"""Render pinned reflection prompts and extract proposed instruction text."""

# https://github.com/gepa-ai/gepa

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, cast

from . import serialization
from .image import ChatMessage, ContentPart, Image
from .reflection_contracts import Signature
from .type_support import override


class _SampleRenderer:
    """Render nested feedback while collecting image attachments in encounter order."""

    def __init__(self) -> None:
        """Start a renderer with no collected images."""
        self.images: list[Image] = []

    def _render_value(self, value: object, level: int = 3) -> str:
        if isinstance(value, Image):
            self.images.append(value)
            return f"[IMAGE-{len(self.images)} — see visual content]\n\n"
        if isinstance(value, dict):
            fields = cast("dict[object, object]", value)
            return self._render_fields(fields.items(), level)
        if isinstance(value, list | tuple):
            items = cast("list[object] | tuple[object, ...]", value)
            fields = {f"Item {index + 1}": item for index, item in enumerate(items)}
            return self._render_fields(fields.items(), level)
        return f"{str(value).strip()}\n\n"

    def _render_fields(
        self,
        fields: Iterable[tuple[object, object]],
        level: int,
    ) -> str:
        text = "".join(
            f"{'#' * level} {key}\n{self._render_value(value, min(level + 1, 6))}"
            for key, value in fields
        )
        return text or "\n"

    def _render_sample(self, sample: Mapping[str, object], number: int) -> str:
        return f"# Example {number}\n" + "".join(
            f"## {key}\n{self._render_value(value)}" for key, value in sample.items()
        )

    def format_samples(self, samples: Sequence[Mapping[str, object]]) -> str:
        """Render all examples with stable markdown numbering and spacing.

        Returns
        -------
        str
            The examples separated by two newlines.

        """
        return "\n\n".join(
            self._render_sample(sample, index + 1)
            for index, sample in enumerate(samples)
        )


class InstructionProposalSignature(Signature):
    """Render feedback and extract candidate instructions using pinned prompts."""

    default_prompt_template = (
        "I provided an assistant with the following instructions to "
        "perform a task for me:\n"
        "```\n"
        "<curr_param>\n"
        "```\n"
        "\n"
        "The following are examples of different task inputs provided to "
        "the assistant along with the assistant's response for each of "
        "them, and some feedback on how the assistant's response could "
        "be better:\n"
        "```\n"
        "<side_info>\n"
        "```\n"
        "\n"
        "Your task is to write a new instruction for the assistant.\n"
        "\n"
        "Read the inputs carefully and identify the input format and "
        "infer detailed task description about the task I wish to solve "
        "with the assistant.\n"
        "\n"
        "Read all the assistant responses and the corresponding "
        "feedback. Identify all niche and domain specific factual "
        "information about the task and include it in the instruction, "
        "as a lot of it may not be available to the assistant in the "
        "future. The assistant may have utilized a generalizable "
        "strategy to solve the task, if so, include that in the "
        "instruction as well.\n"
        "\n"
        "Provide the new instructions within ``` blocks."
    )

    input_keys: ClassVar[list[str]] = [
        "current_instruction_doc",
        "dataset_with_feedback",
        "prompt_template",
    ]
    output_keys: ClassVar[list[str]] = ["new_instruction"]

    @classmethod
    def validate_prompt_template(cls, prompt_template: str | None) -> None:
        """Require both candidate and feedback placeholders in custom templates.

        Raises
        ------
        ValueError
            The supplied template omits a required placeholder.

        """
        if prompt_template is None:
            return
        missing_placeholders = [
            placeholder
            for placeholder in ("<curr_param>", "<side_info>")
            if placeholder not in prompt_template
        ]
        if missing_placeholders:
            error_message = (
                "Missing placeholder(s) in prompt template: "
                f"{', '.join(missing_placeholders)}"
            )
            raise ValueError(
                error_message,
            )

    @classmethod
    @override
    def prompt_renderer(
        cls,
        input_dict: Mapping[str, object],
    ) -> str | list[ChatMessage]:
        """Render text feedback and attach any images to a user message.

        Returns
        -------
        str | list[ChatMessage]
            Plain text or a multimodal message preserving feedback order.

        Raises
        ------
        TypeError
            The instruction, dataset or template has an unsupported type.

        """
        current_instruction = input_dict.get("current_instruction_doc")
        if not isinstance(current_instruction, str):
            error_message = "current_instruction_doc must be a string"
            raise TypeError(error_message)

        dataset = input_dict.get("dataset_with_feedback")
        if not isinstance(dataset, Sequence) or isinstance(dataset, str | bytes):
            error_message = "dataset_with_feedback must be a sequence of records"
            raise TypeError(error_message)

        prompt_template = input_dict.get("prompt_template")
        if prompt_template is None:
            prompt_template = cls.default_prompt_template

        if not isinstance(prompt_template, str):
            message = "prompt_template must be a string"
            raise TypeError(message)
        cls.validate_prompt_template(prompt_template)

        samples = serialization.sequence(
            dataset,
            lambda item: serialization.mapping(
                item,
                serialization.text,
                serialization.identity,
            ),
        )
        renderer = _SampleRenderer()
        formatted_text = renderer.format_samples(samples)
        images = renderer.images

        if images:
            formatted_text = (
                "The evaluation data below includes visual content "
                "("
                f"{len(images)}"
                " image(s)). Analyze both the text and images when "
                "suggesting improvements.\n\n"
            ) + formatted_text

        prompt = prompt_template.replace("<curr_param>", current_instruction)
        prompt = prompt.replace("<side_info>", formatted_text)

        # When images are present, return an OpenAI-compatible multimodal
        # messages list so the reflection LM receives the images inline.
        if images:
            content: list[ContentPart] = [{"type": "text", "text": prompt}]
            content.extend(img.to_openai_content_part() for img in images)
            return [{"role": "user", "content": content}]

        return prompt

    @classmethod
    @override
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        """Extract the proposed instruction from complete or partial code fences.

        Returns
        -------
        dict[str, str]
            The instruction under the canonical output key.

        """

        def extract_instruction_text() -> str:
            # Find the first and last backtick positions (if any)
            start = lm_out.find("```") + 3
            end = lm_out.rfind("```")

            # Handle if the first and last backticks are the same or overlap
            if start >= end:
                # Handle incomplete blocks
                stripped = lm_out.strip()
                if stripped.startswith("```"):
                    # Remove opening ``` and optional language specifier
                    match = re.match(r"^```\S*\n?", lm_out)
                    if match:
                        return lm_out[match.end() :].strip()
                elif stripped.endswith("```"):
                    # Remove closing ```
                    return stripped[:-3].strip()
                return stripped

            # Skip optional language specifier
            content = lm_out[start:end]
            match = re.match(r"^\S*\n", content)
            if match:
                content = content[match.end() :]

            return content.strip()

        return {"new_instruction": extract_instruction_text()}
