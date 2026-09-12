# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from ...core.adapter import RolloutOutput, Trajectory
from ...core.data_loader import DataId
from ...core.state import GEPAState
from ...image import ChatMessage


@runtime_checkable
class CandidateSelector(Protocol):
    def select_candidate_idx(self, state: GEPAState[RolloutOutput, DataId]) -> int: ...


class ReflectionComponentSelector(Protocol):
    def __call__(
        self,
        state: GEPAState[RolloutOutput, DataId],
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]: ...


class LanguageModel(Protocol):
    def __call__(self, prompt: str | list[ChatMessage], /) -> str: ...


@dataclass
class Signature:
    prompt_template: ClassVar[str]
    input_keys: ClassVar[list[str]]
    output_keys: ClassVar[list[str]]

    @classmethod
    def prompt_renderer(
        cls,
        input_dict: Mapping[str, object],
    ) -> str | list[ChatMessage]:
        raise NotImplementedError

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    def run(cls, lm: LanguageModel, input_dict: Mapping[str, object]) -> dict[str, str]:
        full_prompt = cls.prompt_renderer(input_dict)
        lm_res = lm(full_prompt)
        lm_out = lm_res.strip()
        return cls.output_extractor(lm_out)
