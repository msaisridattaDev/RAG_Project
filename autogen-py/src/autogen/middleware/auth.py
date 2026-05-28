"""Authentication middleware — mirrors autogen.net X-LlmQuery-Token validation.

Covers /v1/*, /mcp/*, and /QnA/studypal/* (Program.cs:147, 355).
Accepts the token via:
  - X-LlmQuery-Token header  (REST + MCP)
  - ?token= query string     (WebSocket upgrades — browsers cannot set WS headers)

Uses hmac.compare_digest for timing-safe comparison (prevents timing attacks
against constant-time string equality).

Skips auth for /health, /livez, /readyz, /metrics, /docs, /openapi.json so
liveness probes and API docs remain reachable without credentials.
"""

from __future__ import annotations

import hmac

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from autogen.config.settings import Settings

_PROTECTED_PREFIXES = ("/v1/", "/mcp/", "/QnA/studypal/")

_SKIP_PATHS = frozenset(
    ["/health", "/livez", "/readyz", "/metrics", "/docs", "/openapi.json", "/redoc"]
)


class AuthMiddleware(BaseHTTPMiddleware):
    """Validates X-LlmQuery-Token on all protected routes.

    Protected prefixes: /v1/*, /mcp/*, /QnA/studypal/*
    Skipped paths:      /health, /livez, /readyz, /metrics, /docs, /openapi.json
    """

    def __init__(self, app: ASGIApp, settings: Settings | None = None) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Fast-path: monitoring / docs never require auth
        if path in _SKIP_PATHS or path.startswith("/openapi"):
            return await call_next(request)

        # Only gate protected prefixes; let everything else through
        if not any(path.startswith(p) for p in _PROTECTED_PREFIXES):
            return await call_next(request)

        auth = self._settings.llm_query_auth if self._settings else None
        header_name: str = auth.header_name if auth else "X-LlmQuery-Token"
        expected: str = (auth.allowed_token if auth and auth.allowed_token else "sk-placeholder")

        # Accept token from header (REST/MCP) or ?token= query param (WebSocket)
        token: str | None = (
            request.headers.get(header_name)
            or request.query_params.get("token")
        )

        if not token or not hmac.compare_digest(token, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": f"Invalid or missing {header_name}"},
            )

        request.state.auth_token = token
        return await call_next(request)
