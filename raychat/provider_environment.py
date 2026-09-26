"""Read and save installation-local provider settings without executing shell code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .filesystem import write_bytes
from .provider_settings import provider_settings

if TYPE_CHECKING:
    from collections.abc import Mapping

NAMES = ("RAYCHAT_AUTH_TOKEN", "RAYCHAT_MODEL", "RAYCHAT_BASE_URL")
DEFAULT_FILE = Path(__file__).resolve().parents[1] / "environment" / ".env"


def _parse(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in NAMES or name in values:
            message = f"Invalid or duplicate provider setting on line {number}."
            raise ValueError(message)
        if value.startswith(("'", '"')):
            if len(value) < len("''") or value[-1] != value[0]:
                message = f"Unclosed provider value on line {number}."
                raise ValueError(message)
            value = value[1:-1]
        values[name] = value
    return values


def load(environ: Mapping[str, str], path: Path = DEFAULT_FILE) -> dict[str, str]:
    """Fill missing or blank settings from a UTF-8 file, preserving shell overrides.

    Returns
    -------
    dict[str, str]
        A new environment mapping; the input is never modified.

    Raises
    ------
    ValueError
        The settings file contains malformed entries or invalid UTF-8.

    """
    result = dict(environ)
    if all(result.get(name, "").strip() for name in NAMES):
        return result
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return result
    except UnicodeError:
        message = "Provider settings must be UTF-8 text in environment/.env."
        raise ValueError(message) from None
    for name, value in _parse(text).items():
        if not result.get(name, "").strip():
            result[name] = value
    return result


def save(path: Path, values: Mapping[str, str]) -> None:
    """Validate and atomically save settings with private POSIX permissions."""
    provider_settings(values)
    # Quoting preserves literal #, =, $, backslashes and quote characters. The
    # reader removes only the outer pair; it never interpolates or unescapes.
    text = "".join(f'{name}="{values[name].strip()}"\n' for name in NAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes(path, text.encode("utf-8"))
