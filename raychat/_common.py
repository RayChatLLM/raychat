from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, cast

from raychat.configuration import SETTINGS

from .sdk import (
    Action as Action,
)
from .sdk import (
    ApprovalCallback as ApprovalCallback,
)
from .sdk import (
    CancelCheck as CancelCheck,
)
from .sdk import (
    Chat as Chat,
)
from .sdk import (
    EventCallback as EventCallback,
)
from .sdk import (
    Messages as Messages,
)

DEFAULT_CONTEXT_CHARS = SETTINGS.chat.context_chars

DEFAULT_KEEP_RECENT_TURNS = SETTINGS.chat.keep_recent_turns

DEFAULT_INSTRUCTION_ROLE = SETTINGS.chat.instruction_role

DEFAULT_WORKSPACE = SETTINGS.chat.workspace

_ENVIRONMENT = SETTINGS.chat.environment

_MODEL_ENV = _ENVIRONMENT.model

_CONTEXT_ENV = _ENVIRONMENT.context_chars

_INSTRUCTION_ROLE_ENV = _ENVIRONMENT.instruction_role

MAX_REPLY_CHARS = SETTINGS.limits.max_reply_chars

MAX_PROTOCOL_BYTES = SETTINGS.limits.max_protocol_bytes

MAX_TIMEOUT_SECONDS = SETTINGS.limits.max_timeout_seconds

_BIDI_CONTROLS = frozenset(
    {
        0x061C,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
        0x206A,
        0x206B,
        0x206C,
        0x206D,
        0x206E,
        0x206F,
    },
)

_STRING_CONTROLS = frozenset({0x90, 0x98, 0x9D, 0x9E, 0x9F})

_STRING_ESCAPES = frozenset("PX]^_")

_PROTOCOL_CONFIGURATION = SETTINGS.chat.protocol

RESULT_PREFIX = _PROTOCOL_CONFIGURATION.result_prefix


def _is_positive_finite_number(value: object) -> bool:
    """Return whether *value* is a bounded-duration int or float.

    ``bool`` is intentionally excluded even though it subclasses ``int``.
    Converting very large integers can overflow, so treat those as invalid too.
    """
    if type(value) is not int and type(value) is not float:
        return False
    try:
        return 0 < value <= MAX_TIMEOUT_SECONDS and math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


INSTRUCTION_ROLES = frozenset(SETTINGS.chat.instruction_roles)

MESSAGE_ROLES = frozenset(SETTINGS.chat.message_roles)


def _is_valid_utf8_text(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _detached_callback_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-detached callback value that cannot mutate agent state."""
    return cast("dict[str, Any]", json.loads(json.dumps(value, ensure_ascii=False)))
