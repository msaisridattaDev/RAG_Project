"""Tests for the /health endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from autogen.api.app import create_app
from autogen.config.settings import Settings


@pytest.fixture
def app():
    """Create a test app with known settings."""
    settings = Settings(
        env="test",
        _env_file=None,  # type: ignore[call-arg]
    )
    # Override the auth token for testing
    settings.llm_query_auth.allowed_token = "test-key"
    return create_app(settings=settings)


@pytest.mark.asyncio
async def test_health_returns_ok(app) -> None:
    """GET /health should return 200 with status ok."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "autogen-py"
    assert data["version"] == "0.1.0"
    assert "neetpg" in data["allowed_app_ids"]
    assert "Free" in data["tiers"]
    assert data["auth_header"] == "X-LlmQuery-Token"


@pytest.mark.asyncio
async def test_health_does_not_require_auth(app) -> None:
    """Health endpoint should be accessible without X-LlmQuery-Token."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health", headers={})

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_other_routes_require_auth(app) -> None:
    """Protected routes (/v1/*) should reject requests with missing auth token."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/neetpg/search?q=test")

    assert response.status_code == 401
