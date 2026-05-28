"""LightRag — unified facade for Phase 3 Graph-RAG.

Mirrors autogen.net's LightRag.cs: two public methods (index + query) that
orchestrate the full ingestion pipeline and all four retrieval modes.

Usage::

    rag = LightRag.build(
        app_id="neetpg",
        settings=settings,
        llm=llm_client,
        embedding=embedding_client,
        extraction_model="groq/llama-3.1-70b-versatile",
    )
    await rag.index(documents)
    ctx = await rag.query("what is the MOA of aspirin?")
    print(ctx.build_context_string())
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autogen.logging.setup import get_logger

if TYPE_CHECKING:
    from autogen.config.settings import Settings
    from autogen.models.query import CombinedContext, QueryParam
    from autogen.models.storage import EntityNode, EntityRelation, FullDoc, TextChunk
    from autogen.pipeline.pipeline import EntityExtractionPipeline
    from autogen.protocols.embedding import EmbeddingClient
    from autogen.protocols.llm import LlmClient
    from autogen.retrieval.hybrid import HybridRetrieval

logger = get_logger("autogen.lightrag")


class LightRag:
    """Unified facade for Phase 3 Graph-RAG indexing and querying.

    Two public methods:
        index(docs) → ingest documents into Neo4j + Elasticsearch
        query(q)    → retrieve CombinedContext via any QueryMode

    Construct via ``LightRag.build()`` (wires all dependencies from settings)
    or pass pre-built pipeline and retrieval instances directly.
    """

    def __init__(
        self,
        app_id: str,
        pipeline: EntityExtractionPipeline,
        retrieval: HybridRetrieval,
    ) -> None:
        self._app_id = app_id
        self._pipeline = pipeline
        self._retrieval = retrieval

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        app_id: str,
        settings: Settings,
        llm: LlmClient,
        embedding: EmbeddingClient,
        extraction_model: str,
    ) -> LightRag:
        """Build a fully-wired LightRag instance from application settings.

        This is the single call-site that wires:
            chunker → extractor → normalizer → processor
            → vector_indexer → graph_factory → pipeline
            → keyword_extractor → retrieval → facade

        Args:
            app_id: Tenant identifier (e.g., "neetpg").
            settings: Application settings (used for neo4j, ES, chunk sizes).
            llm: LLM client (already decorated with caching / usage tracking).
            embedding: Embedding client (Jina or compatible).
            extraction_model: Model identifier for extraction + summarization
                (e.g., "groq/llama-3.1-70b-versatile").
        """
        from autogen.chunking.chunker import TextChunker
        from autogen.embeddings.jina import JinaEmbeddingClient
        from autogen.extraction.extractor import EntityExtractor
        from autogen.extraction.keywords import KeywordExtractor
        from autogen.extraction.normalizer import NormalizeNames
        from autogen.extraction.type_resolver import EntityTypeResolver
        from autogen.models.storage import EntityNode, EntityRelation, TextChunk
        from autogen.pipeline.checkpoint import CheckpointManager
        from autogen.pipeline.pipeline import EntityExtractionPipeline
        from autogen.pipeline.processor import EntityRelationProcessor
        from autogen.pipeline.vector_indexer import VectorIndexer
        from autogen.retrieval.hybrid import HybridRetrieval
        from autogen.reranking.reranker import RerankClient
        from autogen.storage.elastic import VectorStoreFactory, _create_es_client
        from autogen.storage.file_kv import FileKvStorage
        from autogen.storage.neo4j_graph import Neo4jGraphStoreFactory

        lr = settings.lightrag
        checkpoint_path = Path(lr.checkpoint_path)
        workspace = checkpoint_path / app_id

        # -- Chunker --
        chunker = TextChunker(
            chunk_token_size=lr.chunk_token_size,
            chunk_overlap_token_size=lr.chunk_overlap_token_size,
            tiktoken_model_name=lr.tiktoken_model_name,
        )

        # -- Extractor stack --
        type_resolver = EntityTypeResolver(llm=llm, model=extraction_model)
        extractor = EntityExtractor(
            llm=llm,
            model=extraction_model,
            type_resolver=type_resolver,
            max_gleaning=lr.entity_extract_max_gleaning,
        )
        normalizer = NormalizeNames(llm=llm, model=extraction_model)

        # -- Checkpoint --
        checker = CheckpointManager(workspace_dir=workspace, app_id=app_id)

        # -- Processor (description summarization) --
        processor = EntityRelationProcessor(
            llm=llm,
            model=extraction_model,
            max_tokens=lr.entity_summary_to_max_tokens,
        )

        # -- Elasticsearch + vector stores --
        es_client = _create_es_client(settings)
        # Cast embedding client — VectorStoreFactory expects JinaEmbeddingClient
        # concretely but any EmbeddingClient is structurally compatible
        jina_client = embedding if isinstance(embedding, JinaEmbeddingClient) else embedding  # type: ignore[assignment]
        vs_factory = VectorStoreFactory(
            es_client=es_client,
            embedding_client=jina_client,
            dim=settings.elasticsearch.embedding_dim,
        )

        # -- Vector indexer --
        indexer = VectorIndexer(
            embedding_client=jina_client,
            store_factory=vs_factory,
        )

        # -- Graph factory --
        graph_factory = Neo4jGraphStoreFactory(
            uri=lr.neo4j_uri,
            user=lr.neo4j_user,
            password=lr.neo4j_password,
        )

        # -- File KV stores --
        kv_base = str(workspace / "kv")
        docs_kv: FileKvStorage = FileKvStorage(
            namespace=f"{app_id}_docs", base_dir=kv_base
        )
        chunks_kv: FileKvStorage = FileKvStorage(
            namespace=f"{app_id}_chunks", base_dir=kv_base
        )
        strings_kv: FileKvStorage = FileKvStorage(
            namespace=f"{app_id}_strings", base_dir=kv_base
        )

        # -- Pipeline --
        pipeline = EntityExtractionPipeline(
            app_id=app_id,
            chunker=chunker,
            extractor=extractor,
            normalizer=normalizer,
            checker=checker,
            processor=processor,
            indexer=indexer,
            graph_factory=graph_factory,
            docs_kv=docs_kv,
            chunks_kv=chunks_kv,
            strings_kv=strings_kv,
            intermediate_dir=workspace / "intermediate",
        )

        # -- Graph store (for querying) --
        graph_store = graph_factory.create(namespace=app_id)

        # -- Per-type vector stores for retrieval --
        chunk_store = vs_factory.create(app_id, TextChunk)
        entity_store = vs_factory.create(app_id, EntityNode)
        relation_store = vs_factory.create(app_id, EntityRelation)

        # -- Keyword extractor --
        kw_extractor = KeywordExtractor(llm=llm, model=extraction_model)

        # -- Reranker (Qwen3-Reranker-4B via settings) --
        rr = settings.reranking_options
        reranker = RerankClient(
            base_url=rr.base_url,
            api_key=rr.api_key,
            model=rr.default_model,
            timeout=rr.timeout_seconds,
        )

        # -- HybridRetrieval --
        retrieval = HybridRetrieval(
            app_id=app_id,
            chunk_store=chunk_store,
            entity_store=entity_store,
            relation_store=relation_store,
            graph_store=graph_store,
            keyword_extractor=kw_extractor,
            reranker=reranker,
        )

        logger.info("lightrag.built", app_id=app_id, model=extraction_model)
        return cls(app_id=app_id, pipeline=pipeline, retrieval=retrieval)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index(self, docs: list[FullDoc]) -> dict[str, int]:
        """Ingest documents into Neo4j + Elasticsearch.

        Runs all 7 pipeline stages with checkpointing. Already-completed
        stages are skipped on re-run.

        Returns:
            Summary counts: {"chunks": N, "entities": M, "relations": K}
        """
        logger.info("lightrag.index.start", app_id=self._app_id, docs=len(docs))
        stats = await self._pipeline.run(docs)
        logger.info("lightrag.index.done", app_id=self._app_id, **stats)
        return stats

    async def query_context(
        self,
        app_id: str,
        query: str,
        params: QueryParam | None = None,
    ) -> CombinedContext:
        """Multi-tenant facade method — Day 19.

        Takes ``app_id`` as first param so callers don't need to know which
        LightRag instance is bound to which exam.  Validates the app_id
        matches the bound exam and delegates to the internal retrieval.

        Args:
            app_id: Tenant / exam identifier.  Logged as a warning if it
                    doesn't match this instance's bound app_id.
            query: Natural language question.
            params: Retrieval knobs (mode, top_k, token budgets).

        Returns:
            CombinedContext — the raw retrieval bundle (no LLM generation).
        """
        if app_id != self._app_id:
            logger.warning(
                "lightrag.query_context.app_id_mismatch",
                expected=self._app_id,
                got=app_id,
            )
        return await self.query(query, params)

    async def query(
        self,
        query: str,
        params: QueryParam | None = None,
    ) -> CombinedContext:
        """Retrieve a CombinedContext for the given query.

        Args:
            query: Natural language question.
            params: Retrieval knobs (mode, top_k, token budgets).
                    Defaults to QueryParam() which uses HYBRID mode.

        Returns:
            CombinedContext with entities, relationships, and source chunks.
        """
        from autogen.models.query import QueryParam as _QP

        if params is None:
            params = _QP()

        logger.debug(
            "lightrag.query",
            app_id=self._app_id,
            mode=params.mode,
            query=query[:80],
        )
        return await self._retrieval.retrieve(query, params)

    async def reset(self, app_id: str) -> None:
        """Dev-only: clear all indices for ``app_id``.

        Deletes the three Elasticsearch indices (entitynode, entityrelation,
        textchunk) and the Neo4j namespace for this exam.  Does NOT touch
        other tenants.  Never call in production.
        """
        if app_id != self._app_id:
            logger.warning(
                "lightrag.reset.app_id_mismatch",
                expected=self._app_id,
                got=app_id,
            )
        logger.warning("lightrag.reset", app_id=self._app_id)
        from autogen.models.storage import EntityNode, EntityRelation, TextChunk

        chunk_store = self._retrieval._chunks
        entity_store = self._retrieval._entities
        relation_store = self._retrieval._relations
        graph_store = self._retrieval._graph

        for store in (chunk_store, entity_store, relation_store):
            try:
                await store.delete_index()
            except Exception:
                logger.warning("lightrag.reset.index_delete_failed", exc_info=True)

        try:
            await graph_store.drop_namespace()
        except Exception:
            logger.warning("lightrag.reset.graph_drop_failed", exc_info=True)
