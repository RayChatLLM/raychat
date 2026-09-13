"""Workspace filesystem plugin registration and explicit public operations."""

from .operations import (
    FILE_COPY_BYTES,
    LIST_PAGE_ENTRIES,
    MAX_FILE_OFFSET,
    OUTPUT_BYTES,
    execute_filesystem,
    register,
    validate_action,
)

__all__ = [
    "FILE_COPY_BYTES",
    "LIST_PAGE_ENTRIES",
    "MAX_FILE_OFFSET",
    "OUTPUT_BYTES",
    "execute_filesystem",
    "register",
    "validate_action",
]
