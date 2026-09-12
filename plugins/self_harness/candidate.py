"""Small prompt overlays and explicitly editable plugin source files."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from raychat.sdk import API_VERSION, PluginContext, workspace_path
from raychat.validation import json_object

from .configuration import SelfHarnessSettings


def parse(
    text: str,
    workspace: Path,
    config: SelfHarnessSettings,
    clusters: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    try:
        value = json_object(text)
    except ValueError:
        # Nano's plain-text response format remains accepted for overlays.
        why, marker, body = text.partition("HARNESS:")
        if (
            not marker
            or not why.strip().startswith("WHY:")
            or not body.rstrip().endswith("END")
        ):
            error_message = "Expected a JSON proposal or WHY:/HARNESS:/END response."
            raise ValueError(
                error_message,
            ) from None
        overlay = body.rstrip()[:-3].strip()
        if overlay.startswith("```") and overlay.endswith("```"):
            overlay = overlay.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        value = {
            "rationale": why.strip()[4:].strip(),
            "overlay": overlay,
            "signature": clusters[0]["signature"],
        }
    if (
        not isinstance(value, dict)
        or set(value) - {"rationale", "signature", "overlay", "files"}
        or not isinstance(value.get("rationale"), str)
        or not value["rationale"].strip()
        or value.get("signature") not in [cluster["signature"] for cluster in clusters]
        or not isinstance(value.get("overlay"), str)
        or not isinstance(value.get("files", {}), dict)
    ):
        error_message = "Proposal requires a rationale, an observed signature, overlay text and optional files."
        raise ValueError(
            error_message,
        )
    if len(value["overlay"].encode("utf-8")) > config.max_overlay_bytes:
        error_message = "Proposed overlay exceeds its byte limit."
        raise ValueError(error_message)
    roots = [workspace_path(workspace, root) for root in config.editable_roots]
    changes = {config.overlay_path: value["overlay"].encode("utf-8")}
    total = 0
    for name, content in value.get("files", {}).items():
        if not isinstance(name, str) or not isinstance(content, str):
            error_message = "Plugin edits must map relative filenames to source text."
            raise ValueError(error_message)
        path = Path(name)
        target = workspace_path(workspace, name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or (path.suffix != ".py" and path.name != "plugin.json")
            or not any(target.is_relative_to(root) for root in roots)
            or "self_harness" in path.parts
        ):
            error_message = (
                "Proposal may edit only Python files inside configured plugin roots."
            )
            raise ValueError(
                error_message,
            )
        if path.name == "plugin.json":
            ctx_manifest = json_object(content)
            if (
                not isinstance(ctx_manifest, dict)
                or ctx_manifest.get("sdk") != API_VERSION
            ):
                error_message = "Expected an SDK v4 plugin manifest."
                raise ValueError(error_message)
        else:
            compile(content, str(target), "exec")
        data = content.encode("utf-8")
        total += len(data)
        if total > config.max_patch_bytes:
            error_message = "Proposed plugin patch exceeds its byte limit."
            raise ValueError(error_message)
        changes[name] = data
    return value, changes


def promote(
    changes: Mapping[str, bytes],
    originals: Mapping[str, bytes | None],
    config: SelfHarnessSettings,
    ctx: PluginContext,
    record: Callable[[str, str], None],
) -> bool:
    """Check for intervening edits, replace atomically, reload, then record acceptance."""
    write = ctx.service("atomic_write")
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
            if originals[name] is None:
                path.unlink(missing_ok=True)
            else:
                write(path, originals[name])
        record("rejected", str(error))

    def commit() -> None:
        record(
            "accepted",
            "Validation passed and the live plugin generation was activated.",
        )
        ctx.notify(
            "Self-harness candidate accepted; the next message uses the new harness.",
        )

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
    return ctx.update_plugins(
        add=added,
        prepare=prepare,
        commit=commit,
        rollback=rollback,
    )
