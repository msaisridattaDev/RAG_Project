"""Shared slowapi rate limiter — imported by app.py and route modules.

Keyed by X-LlmQuery-Token header so each API key gets its own counter,
not its own IP address (important behind reverse proxies).
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address


def _key_by_token(request) -> str:  # type: ignore[no-untyped-def]
    token = request.headers.get("X-LlmQuery-Token") or request.query_params.get("token")
    return token or get_remote_address(request)


limiter = Limiter(key_func=_key_by_token)
