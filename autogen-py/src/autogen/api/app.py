"""FastAPI application factory — mirrors autogen.net's Program.cs / Startup.

Creates and configures the FastAPI app with settings, logging, middleware,
and route registration. Also loads the models.json catalog at startup
(mirroring Program.cs:46-74) and wires singleton services (ES client,
embedding client, reranker, arq pool) via a lifespan handler so they are
reused across requests and properly closed on shutdown.

Phase 5 additions vs Phase 4:
- qna_router + query_router registered (were omitted)
- graph_router registered (new Day 20)
- arq Redis pool wired in lifespan (Day 20)
- Three MCP apps mounted at /mcp/query, /mcp/question, /mcp/user (Day 22)
- slowapi rate limiter wired (Day 22)
- WebSocket route /QnA/studypal/ws registered (Day 21)
"""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from autogen._meta import __app_name__, __version__
from autogen.api.limiter import limiter
from autogen.api.routes.chat import router as chat_router
from autogen.api.routes.health import router as health_router
from autogen.api.routes.qna import router as qna_router
from autogen.api.routes.query import router as query_router
from autogen.api.routes.search import router as search_router
from autogen.config.settings import Settings
from autogen.logging.setup import configure_logging, get_logger
from autogen.middleware.auth import AuthMiddleware
from autogen.di.providers import QnAAgentFactoryFactoryImpl


def _init_singletons(settings: Settings, logger: Any) -> dict[str, Any]:
    """Create the long-lived app singletons from settings.

    Returns a dict keyed by the app.state attribute names used by
    ``get_reference_finder`` and other Depends callables.
    """
    from autogen.embeddings.jina import JinaEmbeddingClient
    from autogen.reranking.reranker import RerankClient
    from autogen.storage.elastic import VectorStoreFactory, _create_es_client

    es_client = _create_es_client(settings)
    embedding_client = JinaEmbeddingClient(settings.embedding_options)
    reranker = RerankClient(
        base_url=settings.reranking_options.base_url,
        api_key=settings.reranking_options.api_key,
        model=settings.reranking_options.default_model,
        timeout=settings.reranking_options.timeout_seconds,
    )
    factory = VectorStoreFactory(
        es_client=es_client,
        embedding_client=embedding_client,
        dim=1024,
    )

    logger.info("lifespan.startup complete")
    return {
        "es_client": es_client,
        "embedding_client": embedding_client,
        "reranker": reranker,
        "store_factory": factory,
    }


async def _shutdown_singletons(state: Any, logger: Any) -> None:
    """Gracefully close all singleton clients."""
    import asyncio

    async def _close(name: str, client: Any) -> None:
        try:
            await client.aclose()
            logger.info("lifespan.shutdown closed", singleton=name)
        except Exception as exc:
            logger.warning("lifespan.shutdown close_failed", singleton=name, error=str(exc))

    singletons: list[tuple[str, Any]] = []
    for attr in ("es_client", "embedding_client", "reranker"):
        client = getattr(state, attr, None)
        if client is not None:
            singletons.append((attr, client))
            setattr(state, attr, None)

    await asyncio.gather(*(_close(name, c) for name, c in singletons))
    logger.info("lifespan.shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional Settings instance. If None, loads from env/.env.

    Returns:
        Configured FastAPI application ready to serve.
    """
    if settings is None:
        settings = Settings()

    log_level = "DEBUG" if settings.env.lower() == "dev" else "INFO"
    configure_logging(log_level)
    logger = get_logger(__name__)
    logger.info(
        "Starting autogen-py",
        version=__version__,
        app_id=settings.app_identity.default_app_id,
    )

    models_path = Path(settings.qna.models_catalog_path)
    if models_path.exists():
        logger.info("models_catalog.loaded", path=str(models_path))
    else:
        logger.warning("models_catalog.missing", path=str(models_path))

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Wire singletons at startup, close them at shutdown."""
        import asyncio

        logger.info("lifespan.startup begin")

        # Core singletons (ES, embedding, reranker, store factory)
        singletons = _init_singletons(settings, logger)
        for key, value in singletons.items():
            setattr(app.state, key, value)

        # LLM stack: UsageTracking → Caching → LiteLLM
        from autogen.llm.builder import build_llm_stack
        from autogen.llm.catalog import create_router
        from autogen.extraction.keywords import KeywordExtractor

        cache_dir = Path(settings.cache.base_path)
        llm = build_llm_stack(
            cache_dir=cache_dir,
            memory_size=settings.cache.memory_size,
            memory_ttl=settings.cache.memory_ttl_seconds,
        )
        app.state.llm = llm
        logger.info("lifespan.llm_stack ready")

        router = create_router(
            settings.qna.tier_configurations,
            settings.qna.models_catalog_path,
        )
        app.state.router = router
        logger.info("lifespan.tier_router ready")

        kw_model = router.model_for("Free", "method_call")
        keyword_extractor = KeywordExtractor(llm=llm, model=kw_model)
        app.state.keyword_extractor = keyword_extractor

        # Conversation store — SQLite (dev) or asyncpg (prod)
        from autogen.conversation.store import SqlConversationStore

        conv_store = SqlConversationStore(
            database_url=settings.conversation.database_url
        )
        await conv_store.init_schema()
        app.state.conv_store = conv_store
        logger.info(
            "lifespan.conv_store ready",
            db=settings.conversation.database_url,
        )

        # Agent factory hierarchy — fully wired
        factory_factory = QnAAgentFactoryFactoryImpl(
            settings=settings,
            llm=llm,
            router=router,
            store_factory=app.state.store_factory,
            embedding_client=app.state.embedding_client,
            reranker=app.state.reranker,
            keyword_extractor=keyword_extractor,
            conv_store=conv_store,
        )
        app.state.factory_factory = factory_factory
        # Expose neo4j_factory at top level so /v1/{app_id}/query can build HybridRetrieval
        app.state.neo4j_factory = factory_factory._graph_factory
        logger.info("lifespan.factory_factory ready")

        # arq Redis pool for background job queue (Day 20)
        arq_pool = None
        try:
            from arq import create_pool
            from arq.connections import RedisSettings as ArqRedisSettings

            arq_pool = await create_pool(ArqRedisSettings.from_dsn(settings.redis.url))
            app.state.arq_pool = arq_pool
            logger.info("lifespan.arq_pool ready", redis_url=settings.redis.url)
        except Exception as exc:
            logger.warning(
                "lifespan.arq_pool failed — ingest endpoints will be unavailable",
                error=str(exc),
            )
            app.state.arq_pool = None

        yield

        # Shutdown
        logger.info("lifespan.shutdown begin")
        await _shutdown_singletons(app.state, logger)
        if app.state.conv_store is not None:
            try:
                await app.state.conv_store.aclose()
                logger.info("lifespan.conv_store closed")
            except Exception as exc:
                logger.warning("lifespan.conv_store close_failed", error=str(exc))
        if arq_pool is not None:
            try:
                await asyncio.wait_for(arq_pool.aclose(), timeout=5.0)
                logger.info("lifespan.arq_pool closed")
            except Exception as exc:
                logger.warning("lifespan.arq_pool close_failed", error=str(exc))

    # ---------------------------------------------------------------------------
    # FastAPI app
    # ---------------------------------------------------------------------------

    app = FastAPI(
        title=__app_name__,
        version=__version__,
        description="Python port of autogen.net",
        lifespan=_lifespan,
    )

    app.state.settings = settings
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Auth middleware — covers /v1/*, /mcp/*, /QnA/studypal/*
    app.add_middleware(AuthMiddleware, settings=settings)

    # ---------------------------------------------------------------------------
    # REST routers
    # ---------------------------------------------------------------------------

    app.include_router(health_router)
    app.include_router(search_router)
    app.include_router(chat_router)
    app.include_router(qna_router)
    app.include_router(query_router)

    # Day 20: graph ingest + query routers (imported lazily to avoid import
    # errors before the graph module is created)
    try:
        from autogen.api.routes.graph import router as graph_router
        app.include_router(graph_router)
        logger.info("graph_router registered")
    except ImportError as exc:
        logger.warning("graph_router not yet available", error=str(exc))

    # ---------------------------------------------------------------------------
    # Day 22: Three MCP servers (Streamable HTTP JSON-RPC POST)
    # ---------------------------------------------------------------------------

    try:
        from autogen.api.mcp.servers import build_query_mcp, build_question_mcp, build_user_mcp

        # MCP servers build their own clients per-call (self-contained);
        # services=None signals that pattern.
        query_mcp = build_query_mcp(None, settings)
        question_mcp = build_question_mcp(None, settings)
        user_mcp = build_user_mcp(None, settings)

        app.mount("/mcp/query", query_mcp.streamable_http_app())
        app.mount("/mcp/question", question_mcp.streamable_http_app())
        app.mount("/mcp/user", user_mcp.streamable_http_app())
        logger.info("mcp servers mounted at /mcp/query /mcp/question /mcp/user")
    except Exception as exc:
        logger.warning("mcp servers not mounted", error=str(exc))

    # ---------------------------------------------------------------------------
    # Day 21: WebSocket endpoint — /QnA/studypal/ws (exact source path)
    # Auth via ?token= query param (browsers cannot set headers on WS upgrades)
    # ---------------------------------------------------------------------------

    @app.websocket("/QnA/studypal/ws")
    async def ws_studypal(websocket: WebSocket):
        from autogen.api.ws.manager import QnAWebSocketManager

        allowed = settings.llm_query_auth.allowed_token or "sk-placeholder"
        token = websocket.query_params.get("token") or websocket.headers.get(
            settings.llm_query_auth.header_name, ""
        )
        if not token or not hmac.compare_digest(token, allowed):
            await websocket.close(code=4001, reason="unauthorized")
            return

        factory_factory = websocket.app.state.factory_factory
        manager = QnAWebSocketManager(factory_factory=factory_factory)
        await manager.handle(websocket, settings=settings)

    return app
