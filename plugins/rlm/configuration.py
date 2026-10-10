"""Budget and trace configuration owned by the rlm plugin.

A :class:`Budget` bundles every numeric limit the recursive language model
obeys: iteration counts, timeouts, clipping sizes and recursion depth.  It
is built once per ``rlm`` tool call by the registration module from three
layers, later first:

1. ``DEFAULT_BUDGET`` - the defaults declared in ``plugin.json`` (mirrored
   here so the plugin is self-describing even without the manifest).
2. plugin settings - operator overrides applied by the host.
3. action arguments - the per-call ``max_iterations`` style overrides sent
   by the calling model.

The loop and the child never see raw settings: they only ever receive a
validated, frozen :class:`Budget` instance plus a resolved trace path.
:class:`TraceSettings` carries the audit-trace policy (on by default,
``rlm_trace.jsonl`` in the workspace root).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from raychat.validation import boolean_field, settings_fields, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_BUDGET: dict[str, int] = {
    "max_iterations": 24,
    "exec_timeout_seconds": 60,
    "total_timeout_seconds": 600,
    "exec_output_chars": 4096,
    "max_llm_calls": 40,
    "max_llm_chunk_chars": 60000,
    "max_llm_reply_chars": 16384,
    "max_depth": 2,
    "max_prompt_bytes": 16777216,
    "max_final_chars": 65536,
    "prompt_preview_chars": 400,
}
"""Default budget values; kept in sync with the ``plugin.json`` defaults."""

FIELD_LIMITS: dict[str, tuple[int, int]] = {
    "max_iterations": (1, 1000),
    "exec_timeout_seconds": (1, 600),
    "total_timeout_seconds": (1, 7200),
    "exec_output_chars": (200, 200000),
    "max_llm_calls": (0, 10000),
    "max_llm_chunk_chars": (1000, 1000000),
    "max_llm_reply_chars": (200, 200000),
    "max_depth": (1, 4),
    "max_prompt_bytes": (0, 1073741824),
    "max_final_chars": (100, 1000000),
    "prompt_preview_chars": (0, 10000),
}
"""Inclusive ``(minimum, maximum)`` for every budget field.

Values below the minimum are rejected (a zero-iteration budget could never
do anything useful, so the caller deserves an explicit error); values above
the maximum are clamped, because an over-large override is harmless once
capped.
"""


class BudgetError(ValueError):
    """Raised when settings or action overrides cannot form a usable budget."""


def _coerce_int(name: str, raw: object) -> int:
    """Coerce one raw setting value into a validated budget integer.

    Parameters
    ----------
    name : str
        Budget field name, used for error messages and limit lookup.
    raw : object
        Raw value taken from plugin settings or action arguments.

    Returns
    -------
    int
        The validated value, clamped to the field's maximum.

    Raises
    ------
    BudgetError
        If the value is not integer-like or is below the field minimum.

    """
    if isinstance(raw, bool):
        message = f"budget field {name!r} must be an int, got bool"
        raise BudgetError(message)
    value: int | None = None
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float) and float(raw).is_integer():
        value = int(raw)
    elif isinstance(raw, str):
        try:
            value = int(raw.strip())
        except ValueError as exc:
            message = f"budget field {name!r} must be an int, got {raw!r}"
            raise BudgetError(message) from exc
    if value is None:
        message = f"budget field {name!r} must be an int, got {raw!r}"
        raise BudgetError(message)
    low, high = FIELD_LIMITS[name]
    if value < low:
        message = f"budget field {name!r} must be >= {low}, got {value}"
        raise BudgetError(message)
    return min(value, high)


@dataclass(frozen=True)
class Budget:
    """Frozen, validated set of limits for one ``rlm`` run.

    Attributes
    ----------
    max_iterations : int
        Maximum number of child ``exec`` rounds (model code executions).
    exec_timeout_seconds : int
        Per-execution timeout for one child ``exec`` round.
    total_timeout_seconds : int
        Timeout for the whole ``rlm`` run, child spawning included.
    exec_output_chars : int
        stdout/stderr/exception clipping length sent back to the model.
    max_llm_calls : int
        Maximum number of ``llm()`` RPCs the child may issue.
    max_llm_chunk_chars : int
        Maximum text length passed into one ``llm()`` call.
    max_llm_reply_chars : int
        Maximum reply length returned from one ``llm()`` call.
    max_depth : int
        Maximum recursion depth; ``1`` disables nested ``rlm()``.
    max_prompt_bytes : int
        Maximum accepted ``prompt``/``prompt_file`` size in bytes.
    max_final_chars : int
        Maximum accepted ``final()`` answer length.
    prompt_preview_chars : int
        Preview length of the symbolic prompt shown in the conversation.

    """

    max_iterations: int
    exec_timeout_seconds: int
    total_timeout_seconds: int
    exec_output_chars: int
    max_llm_calls: int
    max_llm_chunk_chars: int
    max_llm_reply_chars: int
    max_depth: int
    max_prompt_bytes: int
    max_final_chars: int
    prompt_preview_chars: int

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, object] | None,
        overrides: Mapping[str, object] | None = None,
    ) -> Budget:
        """Build a budget from defaults, plugin settings and action overrides.

        Parameters
        ----------
        settings : Mapping[str, object] | None
            Plugin settings as provided by the host (already merged with the
            manifest defaults by the host, but re-merged here for safety).
        overrides : Mapping[str, object] | None, optional
            Per-call overrides from the ``rlm`` action arguments.  Applied
            last so they win over plugin settings.

        Returns
        -------
        Budget
            A frozen budget with every field validated and clamped.
            Any known field holding an unusable value raises
            :class:`BudgetError`; unknown keys in either mapping are
            ignored.

        """
        merged: dict[str, int] = dict(DEFAULT_BUDGET)
        for source in (settings or {}, overrides or {}):
            for name, raw in source.items():
                if name in DEFAULT_BUDGET:
                    merged[name] = _coerce_int(name, raw)
        return cls(**merged)

    def halved(self) -> Budget:
        """Return the budget for one nested ``rlm()`` level.

        Iteration, call and timeout budgets are halved (rounded up, never
        below one) so a recursive worker cannot out-live its parent run.
        Clipping sizes are inherited unchanged.

        Returns
        -------
        Budget
            The budget to pass to the recursive call.

        """
        return replace(
            self,
            max_iterations=math.ceil(self.max_iterations / 2),
            max_llm_calls=math.ceil(self.max_llm_calls / 2),
            exec_timeout_seconds=math.ceil(self.exec_timeout_seconds / 2),
            total_timeout_seconds=math.ceil(self.total_timeout_seconds / 2),
        )

    def to_init_limits(self) -> dict[str, int]:
        """Return the limit fields of the child ``init`` message.

        Returns
        -------
        dict[str, int]
            Mapping with ``output_chars`` and ``chunk_chars`` keys as
            expected by the child program's ``init`` frame.

        """
        return {
            "output_chars": self.exec_output_chars,
            "chunk_chars": self.max_llm_chunk_chars,
        }


@dataclass(frozen=True, kw_only=True)
class TraceSettings:
    """Checked audit-trace policy for one plugin generation.

    Attributes
    ----------
    enabled : bool
        Whether every exec round appends one JSON record to the trace file.
    filename : str
        Workspace-relative trace file name (one JSON object per line).

    """

    enabled: bool
    filename: str

    @classmethod
    def parse(
        cls,
        fields: Mapping[str, object],
        path: str = "rlm",
    ) -> TraceSettings:
        """Validate the trace fields before constructing the immutable record.

        Parameters
        ----------
        fields : Mapping[str, object]
            Plugin settings as provided by the host; missing trace fields
            fall back to tracing ON in ``rlm_trace.jsonl``.
        path : str, optional
            Schema path prefix used in error messages.

        Returns
        -------
        TraceSettings
            Concrete fields detached from mutable configuration input.

        """
        return cls(
            enabled=boolean_field(
                fields.get("trace_enabled", True),
                f"{path}.trace_enabled",
            ),
            filename=text_field(
                fields.get("trace_file", "rlm_trace.jsonl"),
                f"{path}.trace_file",
            ),
        )


_INSTRUCTION_ROLES = frozenset({"system", "user"})


def instruction_role(fields: Mapping[str, object], path: str = "rlm") -> str:
    """Validate the role carrying the sub-model's instruction message.

    Some providers reject or penalize ``system`` messages; ``user`` sends
    the instructions as the opening user turn instead.

    Returns
    -------
    str
        Either ``"system"`` or ``"user"``.

    Raises
    ------
    BudgetError
        If the configured value is neither supported role.

    """
    value = fields.get("instruction_role", "user")
    if not isinstance(value, str) or value not in _INSTRUCTION_ROLES:
        message = f"{path}.instruction_role must be 'system' or 'user'"
        raise BudgetError(message)
    return value


@dataclass(frozen=True, kw_only=True)
class NotesSettings:
    """Checked cross-run API-notes policy for one plugin generation.

    Attributes
    ----------
    enabled : bool
        Whether successful top-level runs persist their code for reuse.
    filename : str
        Workspace-relative notes file name (one JSON document).

    """

    enabled: bool
    filename: str

    @classmethod
    def parse(
        cls,
        fields: Mapping[str, object],
        path: str = "rlm",
    ) -> NotesSettings:
        """Validate the notes fields before constructing the immutable record.

        Parameters
        ----------
        fields : Mapping[str, object]
            Plugin settings as provided by the host; missing notes fields
            fall back to notes ON in ``rlm_api_notes.json``.
        path : str, optional
            Schema path prefix used in error messages.

        Returns
        -------
        NotesSettings
            Concrete fields detached from mutable configuration input.

        """
        return cls(
            enabled=boolean_field(
                fields.get("notes_enabled", True),
                f"{path}.notes_enabled",
            ),
            filename=text_field(
                fields.get("notes_file", "rlm_api_notes.json"),
                f"{path}.notes_file",
            ),
        )


_SETTING_NAMES: tuple[str, ...] = (
    *sorted(DEFAULT_BUDGET),
    "instruction_role",
    "notes_enabled",
    "notes_file",
    "trace_enabled",
    "trace_file",
)


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    fields = settings_fields(raw, "rlm", required=_SETTING_NAMES)
    Budget.from_settings(fields)
    TraceSettings.parse(fields)
    instruction_role(fields)
    NotesSettings.parse(fields)
