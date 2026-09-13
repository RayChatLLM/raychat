"""Typed selection, model and prompt-rendering contracts for reflection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .adapter import RolloutOutput
    from .data_loader import DataId
    from .image import ChatMessage
    from .state import GEPAState


@runtime_checkable
class CandidateSelector(Protocol):
    """Select an existing candidate using its typed optimization history."""

    def select_candidate_idx(self, state: GEPAState[RolloutOutput, DataId]) -> int:
        """Return the selected candidate's index in the optimizer state."""


class ReflectionComponentSelector(Protocol):
    """Select candidate components using the current optimization state."""

    def __call__(
        self,
        state: GEPAState[RolloutOutput, DataId],
        candidate_idx: int,
    ) -> list[str]:
        """Return names of the candidate components to update."""


class LanguageModel(Protocol):
    """Generate text from a plain prompt or typed chat messages."""

    def __call__(self, prompt: str | list[ChatMessage], /) -> str:
        """Return the model's generated text."""


class Signature(ABC):
    """Render checked prompt inputs and extract a structured model response."""

    prompt_template: ClassVar[str]
    input_keys: ClassVar[list[str]]
    output_keys: ClassVar[list[str]]

    @classmethod
    @abstractmethod
    def prompt_renderer(
        cls,
        input_dict: Mapping[str, object],
    ) -> str | list[ChatMessage]:
        """Return a prompt or messages rendered from validated input fields."""

    @classmethod
    @abstractmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        """Return the structured fields extracted from the model response."""

    @classmethod
    def run(cls, lm: LanguageModel, input_dict: Mapping[str, object]) -> dict[str, str]:
        """Run the model on rendered inputs and extract its response fields.

        Returns
        -------
        dict[str, str]
            The signature's structured output fields.

        """
        full_prompt = cls.prompt_renderer(input_dict)
        lm_res = lm(full_prompt)
        return cls.output_extractor(lm_res.strip())
