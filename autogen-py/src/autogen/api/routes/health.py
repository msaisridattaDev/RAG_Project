"""Health check endpoint — mirrors autogen.net /health.

Returns service status, version, allowed app IDs, and available tiers.
"""

from __future__ import annotations

from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, Depends

from autogen._meta import __app_name__, __version__
from autogen.api.deps import get_settings
from autogen.config.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)) -> dict:
    """Return service health status.

    This is the first endpoint to verify during deployment.
    Reports the active app_id list and available tiers for
    sanity-checking deployment configuration.
    """
    return {
        "status": "ok",
        "app_name": __app_name__,
        "version": __version__,
        "allowed_app_ids": settings.app_identity.allowed_app_ids,
        "tiers": ["Free", "Testing", "Regular", "Premium"],
        "auth_header": settings.llm_query_auth.header_name,
    }


@router.get("/livez")
async def livez() -> dict:
    """Kubernetes-style liveness probe — returns 200 when the process is alive."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(settings: Settings = Depends(get_settings)) -> dict:
    """Kubernetes-style readiness probe — returns 200 when Elasticsearch is reachable."""
    es_status = "reachable"
    try:
        es_client = AsyncElasticsearch(hosts=[settings.elasticsearch.url])
        await es_client.ping()
        await es_client.close()
    except Exception:
        es_status = "unreachable"
    return {"elasticsearch": es_status}
