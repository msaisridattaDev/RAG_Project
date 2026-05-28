"""Search endpoint — POST /v1/{app_id}/search.

Exposes the ReferenceFinder pipeline as an HTTP endpoint.
Tenancy is encoded in the URL path (app_id), matching the
/v1/qna/{app_id}/answer pattern used in Phase 5.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from autogen.api.deps import get_settings
from autogen.config.settings import AppIdentitySettings, Settings
from autogen.logging.setup import get_logger
from autogen.models.reference import Reference
from autogen.retrieval.finder import ReferenceFinder

router = APIRouter(tags=["search"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for POST /v1/{app_id}/search."""

    query: str = Field(..., min_length=1, max_length=2000, description="The natural language query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    max_tokens: int = Field(default=4000, ge=100, le=32000, description="Max total tokens across all results")


class SearchResponse(BaseModel):
    """Response body for POST /v1/{app_id}/search."""

    results: list[Reference]
    total: int


# ---------------------------------------------------------------------------
# Dependency: get_reference_finder
# ---------------------------------------------------------------------------


def get_reference_finder(settings: Settings = Depends(get_settings)) -> ReferenceFinder:
    """Resolve a ReferenceFinder instance.

    In Phase 1, this creates a fully wired ReferenceFinder using the
    settings-configured embedding client, reranker, and vector store factory.
    For now, this raises NotImplementedError until the DI wiring is complete.

    TODO: Replace with proper DI container resolution in Phase 2.
    """
    logger = get_logger("autogen.api.search")
    logger.debug("Resolving ReferenceFinder")

    # Lazy imports to avoid circular dependencies at module load time
    from autogen.config.settings import EmbeddingSettings, RerankingSettings
    from autogen.embeddings.jina import JinaEmbeddingClient
    from autogen.reranking.reranker import RerankClient
    from autogen.storage.elastic import VectorStoreFactory

    # Build embedding client from settings
    embedding_opts: EmbeddingSettings = settings.embedding_options
    embedding_client = JinaEmbeddingClient(embedding_opts)

    # Build reranker client from settings
    reranking_opts: RerankingSettings = settings.reranking_options
    reranker = RerankClient(
        base_url=reranking_opts.base_url,
        api_key=reranking_opts.api_key,
        model=reranking_opts.default_model,
        timeout=reranking_opts.timeout_seconds,
    )

    # Build vector store factory (needs ES client — will be wired properly in Phase 2)
    # For now, we raise a clear error if ES is not configured
    from autogen.storage.elastic import _create_es_client

    es_client = _create_es_client(settings)
    factory = VectorStoreFactory(
        es_client=es_client,
        embedding_client=embedding_client,
        dim=1024,
    )

    # Build ReferenceFinder
    finder = ReferenceFinder(
        factory=factory,
        embedding=embedding_client,
        reranker=reranker,
        tiktoken_model=settings.lightrag.tiktoken_model_name,
        overfetch_multiplier=3,
    )
    return finder  # noqa: RET504


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------


@router.post("/v1/{app_id}/search")
async def search(
    app_id: str,
    body: SearchRequest,
    settings: Settings = Depends(get_settings),
    finder: ReferenceFinder = Depends(get_reference_finder),
) -> SearchResponse:
    """Search for references in a specific exam dataset.

    Args:
        app_id: The exam/dataset tenant ID (e.g., "neetpg", "mds").
        body: The search request containing query, top_k, and max_tokens.

    Returns:
        A list of relevant Reference objects sorted by relevance.

    Example:
        curl -X POST localhost:8000/v1/neetpg/search \\
            -H "X-LlmQuery-Token: $TOKEN" \\
            -H "Content-Type: application/json" \\
            -d '{"query": "what reduces inflammation?", "top_k": 5, "max_tokens": 2000}'
    """
    logger = get_logger("autogen.api.search")
    # Validate app_id against the configured allow-list
    identity: AppIdentitySettings = settings.app_identity
    if app_id not in identity.allowed_app_ids:
        logger.warning(
            "search.bad_app_id app_id=%s allowed=%s",
            app_id,
            identity.allowed_app_ids,
        )
        raise HTTPException(status_code=404, detail=f"unknown app_id: {app_id!r}")

    logger.info(
        "search.request app_id=%s query=%s top_k=%d max_tokens=%d",
        app_id,
        body.query[:80],
        body.top_k,
        body.max_tokens,
    )

    results = await finder.find(
        app_id=app_id,
        query=body.query,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
    )

    logger.info("search.ok app_id=%s results=%d", app_id, len(results))

    return SearchResponse(
        results=results,
        total=len(results),
    )
