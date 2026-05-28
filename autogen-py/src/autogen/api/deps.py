"""Reusable FastAPI Depends callables.

Provides dependency injection functions that can be used across all route modules.
Mirrors autogen.net's DI registration pattern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import Request

from autogen.config.settings import Settings
from autogen.di.providers import _cached_settings
from autogen.di.providers import get_http_client as _get_http_client
from autogen.logging.setup import get_logger


def get_settings(request: Request) -> Settings:
    """Retrieve the global Settings instance.

    Prefers ``request.app.state.settings`` (set by ``create_app``) so tests
    can inject custom Settings, and falls back to the process-wide cache
    when called outside a request context.
    """
    state_settings: Settings | None = getattr(request.app.state, "settings", None)
    if state_settings is not None:
        return state_settings
    return _cached_settings()


def get_logger_dep(request: Request) -> Any:  # noqa: ARG001
    """Get a request-scoped structlog logger.

    Usage::

        @router.get("/example")
        async def example(logger = Depends(get_logger_dep)): ...
    """
    return get_logger("autogen.api")


async def get_http_client_dep() -> AsyncIterator[httpx.AsyncClient]:
    """Re-export of the request-scoped HTTP client provider for routes that
    want the FastAPI-managed lifecycle.
    """
    async for client in _get_http_client():
        yield client
