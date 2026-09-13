# https://github.com/gepa-ai/gepa

"""Image wrapper for passing visual data through side_info to reflection VLMs."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict


class ImageURL(TypedDict):
    """Location of one image included in a reflection prompt."""

    url: str


class ImageContentPart(TypedDict):
    """An image part of a multimodal reflection message."""

    type: Literal["image_url"]
    image_url: ImageURL


class TextContentPart(TypedDict):
    """A text part of a multimodal reflection message."""

    type: Literal["text"]
    text: str


ContentPart: TypeAlias = TextContentPart | ImageContentPart


class ChatMessage(TypedDict):
    """A typed text or multimodal message for a reflection model."""

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


_MEDIA_TYPE_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def _guess_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return _MEDIA_TYPE_BY_EXT.get(ext, "image/png")


@dataclass
class Image:
    """Image data for inclusion in ``side_info``, enabling VLM-based reflection.

    Wrap image data in this class and include it as a value (or nested value)
    anywhere inside the ``side_info`` dict returned by your evaluator.  When
    GEPA builds the reflection prompt, ``Image`` objects are automatically
    converted to the OpenAI vision content-part format and passed to the
    reflection LM as inline images.

    This enables a powerful visual feedback loop: your evaluator renders an
    artifact (SVG, 3D model, chart, etc.), passes the rendered image back as
    ASI, and a vision-capable proposer can literally *see* what it's improving.
    The reflection model must support visual input.

    Provide **exactly one** of ``url``, ``path``, or ``base64_data``.

    Args:
        url: A URL pointing to the image, **or** a ``data:`` URI
            (``data:image/png;base64,...``).
        path: A local filesystem path to an image file.  The file is read and
            base64-encoded when the reflection prompt is constructed.
        base64_data: Raw base64-encoded image bytes.  Requires ``media_type``.
        media_type: MIME type (e.g. ``"image/png"``).  Inferred from ``path``
            extension when using *path*; **required** when using *base64_data*.

    Examples::

        # Rendered SVG feedback for visual optimization
        image_b64 = render_svg_to_png(candidate["svg_code"])
        side_info = {
            "RenderedSVG": Image(base64_data=image_b64, media_type="image/png"),
            "Feedback": vlm_feedback,
        }

        # File-based image feedback
        side_info = {
            "Input": "design a logo",
            "RenderedOutput": Image(path="/tmp/logo_v3.png"),
            "Feedback": "The colors are too muted",
        }

    """

    url: str | None = None
    path: str | None = None
    base64_data: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        """Validate the image source and required media type.

        Raises
        ------
        ValueError
            If the source is ambiguous or a base64 image lacks a media type.

        """
        sources = sum(x is not None for x in [self.url, self.path, self.base64_data])
        if sources != 1:
            error_message = "Exactly one of url, path, or base64_data must be provided."
            raise ValueError(
                error_message,
            )
        if self.base64_data is not None and self.media_type is None:
            error_message = "media_type is required when using base64_data."
            raise ValueError(error_message)

    def to_openai_content_part(self) -> ImageContentPart:
        """Convert to an OpenAI-compatible ``image_url`` content-part dict.

        For path-based images the file is read and base64-encoded inline.

        Returns
        -------
        ImageContentPart
            The image URL or data URI in the typed message-content schema.

        Raises
        ------
        ValueError
            If the image source has become invalid after construction.

        """
        if self.url is not None:
            return {"type": "image_url", "image_url": {"url": self.url}}

        if self.path is not None:
            mt = self.media_type or _guess_media_type(self.path)
            data = base64.b64encode(Path(self.path).read_bytes()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mt};base64,{data}"},
            }

        # base64_data
        if self.base64_data is None or self.media_type is None:
            message = "Image source or media type is missing."
            raise ValueError(message)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self.media_type};base64,{self.base64_data}"},
        }
