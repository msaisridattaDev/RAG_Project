"""Graph-aware query endpoint — POST /v1/{app_id}/query.

Phase 3 Day 16. Exposes all four QueryMode paths (NAIVE, LOCAL, GLOBAL, HYBRID)
through a single endpoint, returning a CombinedContext for downstream use
(QnA agent, direct API consumers).

Mirrors autogen.net's LightRag.cs query() surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from autogen.api.deps import get_settings
from autogen.config.settings import Settings
from autogen.logging.setup import get_logger
from autogen.models.enums import QueryMode
from autogen.models.query import CombinedContext, QueryParam

router = APIRouter(tags=["query"])

logger = get_logger("autogen.api.query")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body for POST /v1/{app_id}/query."""

    query: str = Field(..., min_length=1, max_length=4000)
    mode: QueryMode = QueryMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    max_tokens_for_context: int = Field(default=4000, ge=100, le=32000)
    max_tokens_for_entity_context: int = Field(default=2000, ge=100, le=32000)
    max_tokens_for_relation_context: int = Field(default=2000, ge=100, le=32000)
    only_context: bool = Field(
        default=False,
        description="Return the retrieved context bundle only (no LLM generation).",
    )


class QueryResponse(BaseModel):
    """Response body for POST /v1/{app_id}/query."""

    app_id: str
    mode: str
    context: CombinedContext
    context_string: str = Field(
        default="",
        description="Pre-rendered CSV context string ready for prompt injection.",
    )


# ---------------------------------------------------------------------------
# Dependency: build HybridRetrieval from app.state singletons
# ---------------------------------------------------------------------------


def _get_hybrid_retrieval(app_id: str, request: Request, settings: Settings):
    """Build a HybridRetrieval instance for the given app_id.

    Uses the singleton Neo4jGraphStoreFactory and VectorStoreFactory that
    were wired by the lifespan handler in api/app.py.
    """
    from autogen.extraction.keywords import KeywordExtractor
    from autogen.llm.builder import build_llm_stack
    from autogen.models.storage import EntityNode, EntityRelation, TextChunk
    from autogen.retrieval.hybrid import HybridRetrieval
    from autogen.storage.elastic import VectorStoreFactory
    from autogen.storage.neo4j_graph import Neo4jGraphStoreFactory

    neo4j_factory: Neo4jGraphStoreFactory | None = getattr(
        request.app.state, "neo4j_factory", None
    )
    store_factory: VectorStoreFactory | None = getattr(
        request.app.state, "store_factory", None
    )
    if neo4j_factory is None or store_factory is None:
        raise HTTPException(
            status_code=503,
            detail="Graph/vector stores not initialized — check startup logs.",
        )

    # Build LLM client for keyword extraction (uses the chat stack)
    from pathlib import Path
    from autogen.llm.builder import build_llm_stack

    cache_dir = Path(settings.cache.base_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    llm = build_llm_stack(
        cache_dir=cache_dir,
        memory_size=settings.cache.memory_size,
        memory_ttl=settings.cache.memory_ttl_seconds,
    )

    # Resolve the extraction model (cheapest available tier)
    extraction_model = _resolve_extraction_model(settings)

    graph_store = neo4j_factory.create(namespace=app_id)
    chunk_store = store_factory.create(app_id, TextChunk)
    entity_store = store_factory.create(app_id, EntityNode)
    relation_store = store_factory.create(app_id, EntityRelation)
    kw_extractor = KeywordExtractor(llm=llm, model=extraction_model)

    return HybridRetrieval(
        app_id=app_id,
        chunk_store=chunk_store,
        entity_store=entity_store,
        relation_store=relation_store,
        graph_store=graph_store,
        keyword_extractor=kw_extractor,
    )


def _resolve_extraction_model(settings: Settings) -> str:
    """Pick the cheapest configured model for keyword extraction.

    Falls back to a sensible open-source default if no tiers are configured.
    """
    tier_cfgs = settings.qna.tier_configurations
    for tier_name in ("Free", "Testing", "Regular"):
        cfg = tier_cfgs.get(tier_name)
        if cfg:
            return cfg.conversation_model
    return "groq/llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------


@router.post("/v1/{app_id}/query", response_model=QueryResponse)
async def graph_query(
    app_id: str,
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Graph-aware retrieval across NAIVE / LOCAL / GLOBAL / HYBRID modes.

    Args:
        app_id: Tenant/exam dataset identifier (e.g., "neetpg").
        body: Query request — mode, top_k, token budgets.

    Returns:
        CombinedContext carrying entities, relationships, and source chunks,
        plus a pre-rendered context_string for direct prompt injection.

    Example::

        curl -X POST localhost:8000/v1/neetpg/query \\
            -H "X-LlmQuery-Token: $TOKEN" \\
            -H "Content-Type: application/json" \\
            -d '{"query": "MOA of aspirin", "mode": "Hybrid", "top_k": 10}'
    """
    # Validate app_id
    if app_id not in settings.app_identity.allowed_app_ids:
        raise HTTPException(status_code=404, detail=f"unknown app_id: {app_id!r}")

    logger.info(
        "query.request",
        app_id=app_id,
        mode=body.mode,
        query=body.query[:80],
        top_k=body.top_k,
    )

    retrieval = _get_hybrid_retrieval(app_id, request, settings)

    params = QueryParam(
        mode=body.mode,
        top_k=body.top_k,
        max_tokens_for_context=body.max_tokens_for_context,
        max_tokens_for_entity_context=body.max_tokens_for_entity_context,
        max_tokens_for_relation_context=body.max_tokens_for_relation_context,
    )

    ctx = await retrieval.retrieve(body.query, params)

    logger.info(
        "query.ok",
        app_id=app_id,
        mode=body.mode,
        entities=len(ctx.entities),
        relations=len(ctx.relationships),
        sources=len(ctx.sources),
    )

    return QueryResponse(
        app_id=app_id,
        mode=body.mode,
        context=ctx,
        context_string=ctx.build_context_string(),
    )
