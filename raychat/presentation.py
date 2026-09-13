"""Render untrusted console text and open explicitly selected local artifacts."""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO

from raychat._common import (
    _BIDI_CONTROLS,
    _STRING_CONTROLS,
    _STRING_ESCAPES,
    MAX_PROTOCOL_BYTES,
)
from raychat.configuration import SETTINGS

_PARAMETER_BYTES = range(0x30, 0x40)
_INTERMEDIATE_BYTES = range(0x20, 0x30)
_CSI_FINAL_BYTES = range(0x40, 0x7F)
_ESCAPE_FINAL_BYTES = range(0x30, 0x7F)
_SURROGATES = range(0xD800, 0xE000)
_DEL_AND_C1_CONTROLS = range(0x7F, 0xA0)
_SPACE = 0x20
_ESCAPE = 0x1B
_CSI = 0x9B


def _consume_console_csi(value: str, index: int) -> int:
    while index < len(value) and ord(value[index]) in _PARAMETER_BYTES:
        index += 1
    while index < len(value) and ord(value[index]) in _INTERMEDIATE_BYTES:
        index += 1
    if index < len(value) and ord(value[index]) in _CSI_FINAL_BYTES:
        index += 1
    return index


def _consume_console_string(value: str, index: int) -> int:
    while index < len(value):
        codepoint = ord(value[index])
        if codepoint in {0x07, 0x9C}:
            return index + 1
        if codepoint == _ESCAPE and index + 1 < len(value) and value[index + 1] == "\\":
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
    while index < len(value) and ord(value[index]) in _INTERMEDIATE_BYTES:
        index += 1
    if index < len(value) and ord(value[index]) in _ESCAPE_FINAL_BYTES:
        index += 1
    return index


def _visible_character(character: str) -> str:
    codepoint = ord(character)
    if character == "\n":
        return character
    if character == "\t":
        return "    "
    if codepoint in _SURROGATES:
        return "\ufffd"
    if (
        codepoint < _SPACE
        or codepoint in _DEL_AND_C1_CONTROLS
        or codepoint in _BIDI_CONTROLS
    ):
        return ""
    return character


def console_text(value: object, encoding: str | None = None) -> str:
    """Make untrusted text visible without allowing terminal control sequences.

    Returns
    -------
    str
        Sanitized text representable in the requested output encoding.

    """
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    rendered: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        codepoint = ord(character)
        if codepoint == _ESCAPE:
            index = _consume_console_escape(value, index + 1)
            continue
        if codepoint == _CSI:
            index = _consume_console_csi(value, index + 1)
            continue
        if codepoint in _STRING_CONTROLS:
            index = _consume_console_string(value, index + 1)
            continue
        rendered.append(_visible_character(character))
        index += 1
    result = "".join(rendered)
    if encoding:
        try:
            result = result.encode(encoding, "backslashreplace").decode(encoding)
        except LookupError:
            result = result.encode("ascii", "backslashreplace").decode("ascii")
    return result


def open_private_log(path: Path) -> TextIO:
    """Open an append-only UTF-8 log, restricting POSIX permissions to its owner.

    Returns
    -------
    TextIO
        An append stream owned by the caller.

    """
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


def load_protocol(path: Path) -> str:
    """Load a bounded UTF-8 protocol selected explicitly by the operator.

    Returns
    -------
    str
        Nonempty protocol text read from a regular file.

    Raises
    ------
    ValueError
        When the file is oversized, empty, nonregular or contains invalid UTF-8.

    """
    flags = os.O_RDONLY | _file_flag("O_BINARY") | _file_flag("O_NONBLOCK")
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


def _file_flag(name: str) -> int:
    value: object = getattr(os, name, 0)
    if not isinstance(value, int):
        message = "Invalid platform file flag: " + name
        raise TypeError(message)
    return value
