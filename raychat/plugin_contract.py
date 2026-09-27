"""Small package identity contracts shared by configuration and the public SDK."""

API_VERSION = 4


class PluginError(RuntimeError):
    """A plugin declaration, activation, or management operation failed."""
