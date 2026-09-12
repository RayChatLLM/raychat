"""Frontend fixtures use the application parser and explicit plugin resources."""

import argparse
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

from raychat.application import build_runtime
from raychat.entrypoint import _build_parser
from raychat.plugins import Runtime
from raychat.resources import AgentResources
from raychat.sdk import Chat


def arguments(
    argv: Sequence[str] = (),
    *,
    initial_prompt: str | None = None,
) -> argparse.Namespace:
    with (
        tempfile.TemporaryDirectory(prefix="raychat-test-arguments-") as directory,
        mock.patch.object(Path, "home", return_value=Path(directory)),
    ):
        args = _build_parser({}, argv).parse_args(argv)
    args.initial_prompt = initial_prompt
    return args


def resources_fixture() -> AgentResources:
    return AgentResources(Runtime("."), lambda messages: "")


@contextmanager
def provider_fixture(chat: Chat) -> Iterator[None]:
    """Override a registered provider while preserving normal runtime composition."""

    def build(
        workspace: str | Path,
        options: Mapping[str, Any],
        resources: Mapping[str, Any],
    ) -> Runtime:
        runtime = build_runtime(workspace, options, resources)
        runtime.providers["chat_completions"] = lambda args, env: chat
        return runtime

    with (
        tempfile.TemporaryDirectory(prefix="raychat-test-provider-") as directory,
        mock.patch("pathlib.Path.home", return_value=Path(directory)),
        mock.patch("raychat.resources.build_runtime", side_effect=build),
    ):
        yield
