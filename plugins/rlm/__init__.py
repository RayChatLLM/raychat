"""Recursive Language Model (RLM) plugin package.

This package hosts the plugin entrypoint declared in ``plugin.json``
(``__init__:register``) and re-exports the small public surface used by
the host and by the plugin's own modules.  All real behaviour lives in:

``configuration.py``
    Budget and trace-policy parsing shared by registration and the loop.
``child.py``
    The REPL child program as a single string constant (it is shipped to
    the child interpreter via ``python -I -X utf8 -c``, never imported).
``loop.py``
    The async host driver that owns the child process, the protocol, the
    audit trace and the conversation with the sub-model.
``registration.py``
    The ``rlm`` tool definition, validation and execute glue.

The plugin is intentionally a *synchronous* ``rlm`` tool surface: one call
spawns one REPL child.  The child's namespace is built once at init and
persists across exec rounds.  One level of RLM-in-RLM is enabled by
default (``max_depth == 2``); every exec round is audit-traced to
``rlm_trace.jsonl`` and successful top-level runs leave reusable API
notes in ``rlm_api_notes.json``, both in the workspace root unless the
operator disables them.
"""

from .configuration import Budget
from .registration import register

__all__ = ["Budget", "register"]
