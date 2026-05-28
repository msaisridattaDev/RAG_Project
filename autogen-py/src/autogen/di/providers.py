"""FastAPI Depends providers + concrete two-level factory hierarchy.

Mirrors autogen.net's DI registration. Two layers:

1. **Per-request providers** for ``Depends(...)`` callables — settings,
   http client, and the agent factory factory (read from app.state).
2. **Two-level factory delegate hierarchy** as concrete Python classes,
   mirroring autogen.net's::

        QnALlmAgentFactoryFactory(examId) → QnALlmAgentFactory(ctx) → Task<QnALlmAgent>

   rendered here as::

        QnAAgentFactoryFactoryImpl().for_exam(exam_id)
            → QnAAgentFactoryImpl.create(context)
                → QnAAgent (from autogen.agent.qna_agent, fully wired)

``QnAAgentFactoryFactoryImpl`` is constructed once at startup (in the
lifespan handler in ``app.py``) and stored as ``app.state.factory_factory``.
``get_agent_factory_factory`` reads it from there so that FastAPI Depends
callers and WebSocket handlers share the same singleton.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx
from fastapi import Request

from autogen.config.settings import Settings
from autogen.logging.setup import get_logger
from autogen.models.agent import AgentContext
from autogen.protocols.factory import QnAAgentFactory

if TYPE_CHECKING:
    from autogen.agent.qna_agent import QnAAgent
    from autogen.conversation.store import SqlConversationStore
    from autogen.extraction.keywords import KeywordExtractor
    from autogen.llm.catalog import TierModelRouter
    from autogen.protocols.llm import LlmClient
    from autogen.reranking.reranker import RerankClient
    from autogen.retrieval.finder import ReferenceFinder
    from autogen.retrieval.hybrid import HybridRetrieval
    from autogen.storage.elastic import VectorStoreFactory

logger = get_logger("autogen.di.providers")

# ---------------------------------------------------------------------------
# Singleton-style providers (Settings is process-scoped)
# ---------------------------------------------------------------------------


def _cached_settings() -> Settings:
    """Process-wide Settings — returns app.state.settings where possible."""
    return Settings()


# ---------------------------------------------------------------------------
# Per-request providers
# ---------------------------------------------------------------------------


async def get_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Yield a request-scoped ``httpx.AsyncClient``."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


async def get_vector_store(request: Request) -> None:
    """Stub — use ReferenceFinder via get_agent_factory_factory instead."""
    raise NotImplementedError("Use get_agent_factory_factory() → ReferenceFinder instead")


async def get_cache_store(request: Request) -> None:
    """Stub — CacheStore is internal to the LLM decorator stack."""
    raise NotImplementedError("CacheStore is not exposed as a direct Depends")


async def get_graph_store(request: Request) -> None:
    """Stub — use HybridRetrieval via get_agent_factory_factory instead."""
    raise NotImplementedError("Use HybridRetrieval from agent factory instead")


# ---------------------------------------------------------------------------
# Inner factory — one instance per exam_id, caches per-exam stores
# ---------------------------------------------------------------------------


class QnAAgentFactoryImpl:
    """Per-exam inner factory — holds fully-wired singletons.

    Constructed by ``QnAAgentFactoryFactoryImpl.for_exam()`` and cached so
    the per-exam stores (vector, graph) are resolved once per process rather
    than once per request.
    """

    def __init__(
        self,
        exam_id: str,
        llm: LlmClient,
        router: TierModelRouter,
        ref_finder: ReferenceFinder,
        hybrid: HybridRetrieval,
        conv_store: SqlConversationStore,
    ) -> None:
        self._exam_id = exam_id
        self._llm = llm
        self._router = router
        self._ref_finder = ref_finder
        self._hybrid = hybrid
        self._conv_store = conv_store

    async def create(self, context: AgentContext) -> QnAAgent:
        """Return a fully-wired ``QnAAgent`` bound to ``context``."""
        from autogen.agent.qna_agent import QnAAgent as _QnAAgent

        if context.exam_id is None:
            context = context.model_copy(update={"exam_id": self._exam_id})
        return _QnAAgent(
            exam_id=self._exam_id,
            context=context,
            llm=self._llm,
            router=self._router,
            ref_finder=self._ref_finder,
            hybrid=self._hybrid,
            conv_store=self._conv_store,
        )


# ---------------------------------------------------------------------------
# Outer factory — process singleton, builds and caches inner factories
# ---------------------------------------------------------------------------


class QnAAgentFactoryFactoryImpl:
    """Process-singleton outer factory — creates per-exam inner factories.

    Constructed once in the lifespan handler (``app.py``) after all infra
    singletons (ES, embeddings, reranker, LLM stack) are ready.  Stored at
    ``app.state.factory_factory`` and retrieved via ``get_agent_factory_factory``.

    ``for_exam()`` is idempotent: the first call for a given ``exam_id``
    builds ``HybridRetrieval``, ``ReferenceFinder``, and ``Neo4jGraphStore``
    for that exam and caches them.  Subsequent calls return the cached factory.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        llm: LlmClient,
        router: TierModelRouter,
        store_factory: VectorStoreFactory,
        embedding_client: object,
        reranker: RerankClient,
        keyword_extractor: KeywordExtractor,
        conv_store: SqlConversationStore,
    ) -> None:
        from autogen.storage.neo4j_graph import Neo4jGraphStoreFactory

        self._settings = settings
        self._llm = llm
        self._router = router
        self._store_factory = store_factory
        self._embedding_client = embedding_client
        self._reranker = reranker
        self._kw = keyword_extractor
        self._conv_store = conv_store
        self._factories: dict[str, QnAAgentFactoryImpl] = {}

        lg = settings.lightrag
        self._graph_factory = Neo4jGraphStoreFactory(
            uri=lg.neo4j_uri,
            user=lg.neo4j_user,
            password=lg.neo4j_password,
        )

    def for_exam(self, exam_id: str) -> QnAAgentFactory:
        """Return (building on first call) the inner factory for ``exam_id``."""
        if exam_id not in self._factories:
            self._factories[exam_id] = self._build_factory(exam_id)
            logger.info("factory.built", exam_id=exam_id)
        return self._factories[exam_id]

    def _build_factory(self, exam_id: str) -> QnAAgentFactoryImpl:
        """Construct per-exam stores and wrap them in a ``QnAAgentFactoryImpl``."""
        from autogen.models.storage import EntityNode, EntityRelation, TextChunk
        from autogen.retrieval.finder import ReferenceFinder
        from autogen.retrieval.hybrid import HybridRetrieval

        graph_store = self._graph_factory.create(exam_id)

        chunk_store = self._store_factory.create(exam_id, TextChunk)
        entity_store = self._store_factory.create(exam_id, EntityNode)
        relation_store = self._store_factory.create(exam_id, EntityRelation)

        ref_finder = ReferenceFinder(
            factory=self._store_factory,
            embedding=self._embedding_client,
            reranker=self._reranker,
        )

        hybrid = HybridRetrieval(
            app_id=exam_id,
            chunk_store=chunk_store,
            entity_store=entity_store,
            relation_store=relation_store,
            graph_store=graph_store,
            keyword_extractor=self._kw,
            reranker=self._reranker,
        )

        return QnAAgentFactoryImpl(
            exam_id=exam_id,
            llm=self._llm,
            router=self._router,
            ref_finder=ref_finder,
            hybrid=hybrid,
            conv_store=self._conv_store,
        )


# ---------------------------------------------------------------------------
# FastAPI Depends callable — reads singleton from app.state
# ---------------------------------------------------------------------------


def get_agent_factory_factory(request: Request) -> QnAAgentFactoryFactoryImpl:
    """Return the process-singleton ``QnAAgentFactoryFactoryImpl`` from app.state.

    Usage in a FastAPI route::

        async def handler(
            app_id: str,
            ff: QnAAgentFactoryFactoryImpl = Depends(get_agent_factory_factory),
        ):
            factory = ff.for_exam(app_id)
            agent   = await factory.create(ctx)
            ...

    Raises ``RuntimeError`` if called before the lifespan handler has run
    (should never happen in normal operation).
    """
    ff = getattr(request.app.state, "factory_factory", None)
    if ff is None:
        raise RuntimeError(
            "factory_factory not initialized — ensure lifespan startup completed"
        )
    return ff
