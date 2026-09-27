"""HTTP-only redirect policy, loaded when a transport is first requested."""

from urllib.request import HTTPRedirectHandler

from raychat.type_support import override


class NoRedirects(HTTPRedirectHandler):
    """Avoid forwarding credentials to a redirect target."""

    @staticmethod
    @override
    def redirect_request(*_args: object, **_kwargs: object) -> None:
        """Reject every redirect without forwarding request credentials."""
        return
