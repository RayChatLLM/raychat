"""Version-one UI state transfer, separate from rendering caches and live workers."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from raychat.handoff import (
    document,
    editor_parts,
    editor_state,
    export_plugins,
    restore_plugins,
)
from raychat.storage import SessionStore
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

if TYPE_CHECKING:
    from .controller import ChatView, _TuiController

_POINT_DIMENSIONS = 2


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


def capture_view(owner: ChatView) -> dict[str, object]:
    """Detach one chat's frontend state before dispatch or process handoff.

    Returns
    -------
    dict[str, object]
        Conversation-local UI fields without a live worker reference.

    """
    selection = owner.selection
    return {
        "state": owner.state.handoff,
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
            "rows": selection.rows,
            "width": selection.width,
            "dragging": selection.dragging,
            "pointer": selection.pointer,
        },
    }


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
) -> dict[str, object]:
    """Capture all idle conversations, drafts, queue transactions and navigation.

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
        root.export_snapshot()
        if root is not None
        else store.snapshot()
        if store is not None
        else {"history": [], "state": resources.runtime.state}
    )
    try:
        plugins = export_plugins(resources.runtime)
    except Exception as error:
        if strict:
            raise
        plugins = {"unavailable": str(error)}
    picker = controller.picker
    live = resources.live
    return document({
        "version": VERSION,
        "session": snapshot,
        "store": writer(controller),
        "plugins": plugins,
        "sources": resources.runtime.export_sources(),
        "views": {key: capture_view(value) for key, value in controller.views.items()},
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
    })


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
