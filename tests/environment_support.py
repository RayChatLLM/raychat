"""Explicit, synthetic provider configuration for isolated offline checks."""

from __future__ import annotations


def provider_environment(
    *,
    url: str = "http://127.0.0.1:1/v1",
    model: str = "test",
) -> dict[str, str]:
    """Supply every required variable without reading the operator's credentials.

    Returns
    -------
    dict[str, str]
        A detached environment mapping pointing only at the selected fixture.

    """
    return {
        "RAYCHAT_AUTH_TOKEN": "fixture-token",
        "RAYCHAT_MODEL": model,
        "RAYCHAT_BASE_URL": url,
    }
