from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from raychat._common import (
    _BIDI_CONTROLS,
    _STRING_CONTROLS,
    _STRING_ESCAPES,
    MAX_PROTOCOL_BYTES,
)
from raychat.configuration import SETTINGS


def _consume_console_csi(value: str, index: int) -> int:
    while index < len(value) and 0x30 <= ord(value[index]) <= 0x3F:
        index += 1
    while index < len(value) and 0x20 <= ord(value[index]) <= 0x2F:
        index += 1
    if index < len(value) and 0x40 <= ord(value[index]) <= 0x7E:
        index += 1
    return index


def _consume_console_string(value: str, index: int) -> int:
    while index < len(value):
        codepoint = ord(value[index])
        if codepoint in {0x07, 0x9C}:
            return index + 1
        if codepoint == 0x1B and index + 1 < len(value) and value[index + 1] == "\\":
            return index + 2
        index += 1
    return len(value)


def _consume_console_escape(value: str, index: int) -> int:
    if index >= len(value):
        return index
    introducer = value[index]
    if introducer == "[":
        return _consume_console_csi(value, index + 1)
    if introducer in _STRING_ESCAPES:
        return _consume_console_string(value, index + 1)
    while index < len(value) and 0x20 <= ord(value[index]) <= 0x2F:
        index += 1
    if index < len(value) and 0x30 <= ord(value[index]) <= 0x7E:
        index += 1
    return index


def _console_text(value: object, encoding: str | None = None) -> str:
    """Make untrusted text visible without allowing terminal control sequences."""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    rendered: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        codepoint = ord(character)
        if codepoint == 0x1B:
            index = _consume_console_escape(value, index + 1)
            continue
        if codepoint == 0x9B:
            index = _consume_console_csi(value, index + 1)
            continue
        if codepoint in _STRING_CONTROLS:
            index = _consume_console_string(value, index + 1)
            continue
        if character == "\n":
            rendered.append(character)
        elif character == "\t":
            rendered.append("    ")
        elif 0xD800 <= codepoint <= 0xDFFF:
            rendered.append("\ufffd")
        elif (
            codepoint < 0x20 or 0x7F <= codepoint <= 0x9F or codepoint in _BIDI_CONTROLS
        ):
            pass
        else:
            rendered.append(character)
        index += 1
    result = "".join(rendered)
    if encoding:
        try:
            result = result.encode(encoding, "backslashreplace").decode(encoding)
        except LookupError:
            result = result.encode("ascii", "backslashreplace").decode("ascii")
    return result


def _command_text(action: Mapping[str, Any] | None) -> str:
    """Return complete ASCII JSON for the argv that shell=False will execute."""
    if not isinstance(action, Mapping):
        return "<invalid command action>"
    argv = action.get("argv")
    if (
        not isinstance(argv, (list, tuple))
        or not argv
        or not all(isinstance(item, str) for item in argv)
    ):
        return "<invalid argv>"
    cwd = action.get("cwd", ".")
    if not isinstance(cwd, str):
        return "<invalid cwd>"
    return (
        json.dumps(list(argv), ensure_ascii=True, separators=(",", ":"))
        + "  cwd="
        + json.dumps(cwd, ensure_ascii=True)
    )


def _open_private_log(path: Path) -> TextIO:
    """Open an append-only UTF-8 log, restricting POSIX permissions to its owner."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    file_mode = SETTINGS.storage.file_mode
    fd = os.open(path, flags, file_mode)
    try:
        if os.name == "posix":
            os.fchmod(fd, file_mode)
        return os.fdopen(fd, "a", encoding="utf-8", newline="\n")
    except BaseException:
        os.close(fd)
        raise


def _load_protocol(path: Path) -> str:
    """Load a bounded UTF-8 protocol selected explicitly by the operator."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            error_message = "Protocol file must be a regular file."
            raise ValueError(error_message)
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            data = stream.read(MAX_PROTOCOL_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(data) > MAX_PROTOCOL_BYTES:
        error_message = (
            f"Protocol file exceeds the {MAX_PROTOCOL_BYTES}-byte size limit."
        )
        raise ValueError(
            error_message,
        )
    try:
        protocol = data.decode("utf-8")
    except UnicodeDecodeError:
        error_message = "Protocol file must be valid UTF-8."
        raise ValueError(error_message) from None
    if not protocol.strip():
        error_message = "Protocol file must contain nonempty text."
        raise ValueError(error_message)
    return protocol
