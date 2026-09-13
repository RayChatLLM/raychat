"""Frontend fixtures use the application parser and explicit plugin resources."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.application import build_runtime
from raychat.entrypoint import build_parser
from raychat.plugins import Runtime
from raychat.resources import AgentResources
from raychat.validation import configuration_fields

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator, Mapping, Sequence

    from raychat.sdk import Chat


def argument_fields(args: argparse.Namespace) -> Mapping[str, object]:
    """Expose parsed plugin arguments as checked names with unknown values.

    Returns
    -------
    Mapping[str, object]
        The real parser values without propagating Namespace's dynamic types.

    """
    raw: object = vars(args)
    return configuration_fields(raw, "parsed arguments")


def arguments(
    argv: Sequence[str] = (),
    *,
    initial_prompt: str | None = None,
) -> argparse.Namespace:
    """Parse application options while isolating the operator's home directory.

    Returns
    -------
    argparse.Namespace
        Real parser results with the requested initial prompt.

    """
    with (
        tempfile.TemporaryDirectory(prefix="raychat-test-arguments-") as directory,
        mock.patch.object(Path, "home", return_value=Path(directory)),
    ):
        args = build_parser({}, argv).parse_args(argv)
    args.initial_prompt = initial_prompt
    return args


def resources_fixture() -> AgentResources:
    """Create minimal application resources for frontend checks.

    Returns
    -------
    AgentResources
        An empty runtime with a deterministic chat callback.

    """
    return AgentResources(Runtime("."), lambda _messages: "")


@contextmanager
def provider_fixture(chat: Chat) -> Iterator[None]:
    """Override a registered provider while preserving normal runtime composition."""

    def build(
        workspace: str | Path,
        options: Mapping[str, object],
        resources: Mapping[str, object],
    ) -> Runtime:
        runtime = build_runtime(workspace, options, resources)
        runtime.providers["chat_completions"] = lambda _args, _env: chat
        return runtime

    with (
        tempfile.TemporaryDirectory(prefix="raychat-test-provider-") as directory,
        mock.patch("pathlib.Path.home", return_value=Path(directory)),
        mock.patch("raychat.resources.build_runtime", side_effect=build),
    ):
        yield
