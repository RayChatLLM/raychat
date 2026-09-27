"""Version-one UI state transfer, separate from rendering caches and live workers."""

from __future__ import annotations

import base64
import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from raychat.checkpoint_stream import PluginFragment
from raychat.checkpoint_stream import snapshot as stream_snapshot
from raychat.handoff import (
    document,
    editor_parts,
    editor_state,
    export_plugins,
    restore_plugins,
    stream_plugins,
)
from raychat.sdk import checkpoint_snapshot
from raychat.storage import SessionStore
from raychat.type_support import override
from raychat.validation import (
    array_field,
    boolean_field,
    configuration_fields,
    integer_field,
    text_field,
)
from raychat_bootstrap.wire import VERSION

from .picker import Choice, Picker
from .selection import TextSelection
from .state import _PrefixTranscriptRows

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .controller import ChatView, _TuiController

_POINT_DIMENSIONS = 2


class _DeferredViews(Mapping[str, object]):
    """Capture one owned view at a time during the immediate synchronous send."""

    def __init__(self, views: Mapping[str, ChatView]) -> None:
        self._owners = dict(views)

    @override
    def __getitem__(self, key: str) -> object:
        return capture_view(self._owners[key], compact_state=True)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._owners)

    @override
    def __len__(self) -> int:
        return len(self._owners)


def _point(value: object, *, minimum: int | None = 0) -> tuple[int, int] | None:
    if value is None:
        return None
    data = array_field(value, "selection point")
    if len(data) != _POINT_DIMENSIONS:
        message = "A selection point requires two coordinates."
        raise ValueError(message)
    return integer_field(data[0], "row", minimum=minimum), integer_field(
        data[1],
        "column",
        minimum=minimum,
    )


def capture_view(
    owner: ChatView,
    *,
    include_state: bool = True,
    compact_state: bool = False,
) -> dict[str, object]:
    """Detach one chat's frontend state before dispatch or process handoff.

    Returns
    -------
    dict[str, object]
        Conversation-local UI fields without a live worker reference.

    """
    selection = owner.selection
    lazy_selection = selection.needs_rebind or isinstance(
        selection.rows,
        _PrefixTranscriptRows,
    )
    selection_history = (
        selection.expected_history
        if selection.needs_rebind
        else cast("_PrefixTranscriptRows", selection.rows).fingerprint
        if lazy_selection
        else ""
    )
    view: dict[str, object] = {
        "editor": editor_state(owner.editor.text, owner.editor.cursor),
        "queue": owner.message_queue.export_handoff(),
        "input_history": owner.input_history.export_handoff(),
        "scroll": owner.scroll_offset,
        "completion": {
            "selected": owner.completion.selected,
            "dismissed": owner.completion.dismissed,
        },
        "selection": {
            "anchor": selection.anchor,
            "focus": selection.focus,
            "rows": () if lazy_selection else selection.rows,
            "lazy": lazy_selection,
            "history": selection_history,
            "width": selection.width,
            "dragging": selection.dragging,
            "pointer": selection.pointer,
        },
    }
    if include_state:
        view["state"] = (
            owner.state.compact_handoff() if compact_state else owner.state.handoff
        )
    return view


def restore_selection(value: object) -> TextSelection:
    """Restore a held pointer with a fresh clock, accepting older saved selections.

    Returns
    -------
    TextSelection
        The captured display coordinates, with no inherited monotonic deadline.

    """
    selection = configuration_fields(value, "selection")
    restored = TextSelection(
        anchor=_point(selection["anchor"]),
        focus=_point(selection["focus"]),
        rows=tuple(
            text_field(row, "selection row", allow_empty=True)
            for row in array_field(selection["rows"], "rows")
        ),
        width=integer_field(selection["width"], "width", minimum=0),
        dragging=boolean_field(selection["dragging"], "dragging"),
        pointer=_point(selection.get("pointer"), minimum=None),
        needs_rebind=boolean_field(selection.get("lazy", False), "lazy selection"),
        expected_history=text_field(
            selection.get("history", ""),
            "selection history",
            allow_empty=True,
        ),
    )
    if not restored.dragging:
        restored.finish()
    return restored


def writer(controller: _TuiController) -> dict[str, object] | None:
    """Describe the current root journal before work receives dispatch ownership.

    Returns
    -------
    dict[str, object] | None
        The current durable journal identity, or no persistence backend.

    Raises
    ------
    TypeError
        The persistence backend cannot transfer writer ownership.

    """
    root = controller.views[controller.root_id].worker.session
    store = controller.resources.store if root is None else root.store
    if store is None:
        return None
    if not isinstance(store, SessionStore):
        message = "This persistence backend has no process writer handoff."
        raise TypeError(message)
    return {
        "id": store.session_id,
        "directory": str(store.directory),
        "committed": store.committed,
    }


def capture(
    controller: _TuiController,
    *,
    strict: bool = True,
    defer_views: bool = False,
) -> dict[str, object]:
    """Capture all idle conversations, drafts, queue transactions and navigation.

    Deferred views retain their owners, not snapshots; plugin values are also
    borrowed until serialization. The caller must consume them synchronously
    before the frontend, workers or plugin resources can change.

    Returns
    -------
    dict[str, object]
        The complete versioned process handoff.

    Raises
    ------
    ValueError
        A persistence backend has no process writer handoff.

    """
    resources = controller.resources
    root = controller.views[controller.root_id].worker.session
    store = resources.store if root is None else root.store
    if store is not None and not isinstance(store, SessionStore):
        message = "This persistence backend has no process writer handoff."
        raise ValueError(message)
    snapshot = (
        (stream_snapshot(root) if defer_views else checkpoint_snapshot(root))
        if root is not None
        else store.snapshot()
        if store is not None
        else {"history": [], "state": copy.deepcopy(resources.runtime.state)}
    )
    try:
        plugins: object = (
            PluginFragment(lambda: stream_plugins(resources.runtime), strict)
            if defer_views
            else export_plugins(resources.runtime)
        )
    except Exception as error:
        if strict:
            raise
        plugins = {"unavailable": str(error)}
    picker = controller.picker
    live = resources.live
    # Deferred views create containers on access so the transport can release
    # one before the next. Borrowed plugin values are serialized without mutation.
    return {
        "version": VERSION,
        "session": snapshot,
        "store": writer(controller),
        "plugins": plugins,
        "sources": copy.deepcopy(resources.runtime.export_sources()),
        "views": _DeferredViews(controller.views)
        if defer_views
        else {
            key: capture_view(value, compact_state=True)
            for key, value in controller.views.items()
        },
        "root": controller.root_id,
        "focus": controller.focused_id,
        "show_system": controller.show_system,
        "decoder": controller.decoder.export_handoff(),
        "pending_input": ""
        if live is None
        else base64.b64encode(live.input).decode("ascii"),
        "menu": controller.menu_name,
        "picker": None
        if picker is None
        else {
            "title": picker.title,
            "choices": [{"id": c.id, "label": c.label} for c in picker.all_choices],
            "searchable": picker.searchable,
            "query": picker.query,
            "index": picker.index,
            "offset": picker.offset,
        },
    }


def restore(controller: _TuiController, value: object) -> None:
    """Reconstruct the frontend before acknowledging readiness to the supervisor.

    Raises
    ------
    ValueError
        The restored navigation graph differs from the captured graph.

    """
    data = document(value)
    controller.view.worker.restore_conversation(
        configuration_fields(data["session"], "session"),
    )
    restore_plugins(controller.resources.runtime, data["plugins"])
    controller.sync_handoff_chats()
    views = configuration_fields(data["views"], "chat views")
    if views.keys() != controller.views.keys() or data["root"] != controller.root_id:
        message = "Candidate navigation does not match the handoff."
        raise ValueError(message)
    for key, raw in views.items():
        saved = configuration_fields(raw, "chat view")
        owner = controller.views[key]
        owner.active_job_id = None
        owner.state.handoff = saved["state"]
        owner.editor.set_text(*editor_parts(saved["editor"]))
        owner.message_queue.restore_handoff(saved["queue"])
        owner.input_history.restore_handoff(saved.get("input_history"))
        owner.scroll_offset = integer_field(saved["scroll"], "scroll", minimum=0)
        completion = configuration_fields(saved["completion"], "completion")
        owner.completion.selected = integer_field(
            completion["selected"],
            "completion selection",
            minimum=0,
        )
        owner.completion.dismissed = text_field(
            completion["dismissed"],
            "dismissed",
            nullable=True,
        )
        owner.selection = restore_selection(saved["selection"])
    focus = text_field(data["focus"], "focus")
    if focus not in controller.views:
        message = "Missing focused chat."
        raise ValueError(message)
    controller.activate_handoff_chat(focus)
    controller.show_system = boolean_field(data["show_system"], "show system")
    controller.decoder.restore_handoff(data["decoder"])
    if controller.resources.live is not None:
        controller.resources.live.input.extend(
            base64.b64decode(
                text_field(data["pending_input"], "pending input", allow_empty=True),
                validate=True,
            ),
        )
    controller.menu_name = text_field(data["menu"], "menu", nullable=True)
    if data["picker"] is not None:
        picker = configuration_fields(data["picker"], "picker")
        choices = [
            configuration_fields(raw, "choice")
            for raw in array_field(picker["choices"], "choices")
        ]
        controller.picker = Picker(
            text_field(picker["title"], "picker title"),
            [
                Choice(
                    text_field(c["id"], "choice id"),
                    text_field(c["label"], "choice label"),
                )
                for c in choices
            ],
            searchable=boolean_field(picker.get("searchable", False), "searchable"),
        )
        controller.picker.query = text_field(
            picker.get("query", ""),
            "filter",
            allow_empty=True,
        )
        controller.picker.replace(controller.picker.all_choices)
        controller.picker.index = integer_field(
            picker["index"],
            "picker index",
            minimum=0,
        )
        controller.picker.offset = integer_field(
            picker["offset"],
            "picker offset",
            minimum=0,
        )
