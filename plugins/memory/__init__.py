"""Register persistent memory with explicit cross-plugin service contracts."""

from .configuration import MemorySettings as MemorySettings
from .registration import register as register
from .registration import validate_action as validate_action
from .store import MAX_MEMORY_CHARS as MAX_MEMORY_CHARS
from .store import MAX_MEMORY_CONTEXT_CHARS as MAX_MEMORY_CONTEXT_CHARS
from .store import MAX_MEMORY_FILE_BYTES as MAX_MEMORY_FILE_BYTES
from .store import MAX_MEMORY_ID as MAX_MEMORY_ID
from .store import MAX_MEMORY_ITEMS as MAX_MEMORY_ITEMS
from .store import MemoryStore as MemoryStore
