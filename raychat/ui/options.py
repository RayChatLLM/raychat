"""Decode command-line display settings before entering the interactive loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from raychat.validation import (
    boolean_field,
    configuration_fields,
    integer_field,
    number_field,
    text_field,
)

if TYPE_CHECKING:
    import argparse


@dataclass(kw_only=True)
class TuiOptions:
    """Concrete display and interaction settings consumed by the UI thread."""

    ascii: bool
    no_animation: bool
    color_256: bool
    fps: float
    quality: int
    max_steps: int
    initial_prompt: str | None
    model: str | None
    provider: str
    workspace: str

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> TuiOptions:
        """Validate parser output once before rendering or processing input.

        Returns
        -------
        TuiOptions
            Typed values detached from the parser's dynamic namespace.

        """
        raw: object = vars(namespace)
        fields = configuration_fields(raw, "terminal options")
        return cls(
            ascii=boolean_field(fields.get("ascii"), "ascii"),
            no_animation=boolean_field(fields.get("no_animation"), "no_animation"),
            color_256=boolean_field(fields.get("color_256"), "color_256"),
            fps=number_field(fields.get("fps"), "fps"),
            quality=integer_field(fields.get("quality"), "quality", minimum=0),
            max_steps=integer_field(fields.get("max_steps"), "max_steps", minimum=0),
            initial_prompt=text_field(
                fields.get("initial_prompt"),
                "initial_prompt",
                nullable=True,
            ),
            model=text_field(fields.get("model"), "model", nullable=True),
            provider=text_field(fields.get("provider"), "provider"),
            workspace=text_field(fields.get("workspace"), "workspace"),
        )
