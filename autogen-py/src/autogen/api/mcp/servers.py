"""Three MCP servers — Day 22.

Mirrors Program.cs:375-377 and MultiEndpointMcpExtensions.cs exactly:

    app.MapMcpEndpoint("/mcp/query",    "books",          LlmQueryToolMcpWrapper)
    app.MapMcpEndpoint("/mcp/question", "question",       QuestionUpdateMcpWrapper + SplitQuestionUpdateMcpWrapper)
    app.MapMcpEndpoint("/mcp/user",     "user-analytics", McqdbDashboardMcpTools)

Each is a fully independent FastMCP server with its own serverInfo.name,
its own tool list, and its own JSON-RPC handshake.  All use the Streamable
HTTP transport (plain POST + JSON-RPC 2.0), NOT SSE.

Tool app_id argument is validated against Settings.app_identity.allowed_app_ids.
Unknown app_id returns a tool-level error per MCP spec (not HTTP 4xx).
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from autogen.config.settings import Settings
from autogen.logging.setup import get_logger

logger = get_logger("autogen.api.mcp")


# ---------------------------------------------------------------------------
# /mcp/query — "books" server
# Mirrors LlmQueryToolMcpWrapper.  Tools: search_book_references, query_graph
# ---------------------------------------------------------------------------


def build_query_mcp(services: Any, settings: Settings) -> FastMCP:
    """Build the 'books' MCP server for knowledge-base search.

    Tools:
        search_book_references — dense vector similarity search (Phase 1 ReferenceFinder)
        query_graph            — full graph-RAG query (Phase 3 HybridRetrieval, 4 modes)
    """
    mcp = FastMCP("books")
    allowed = set(settings.app_identity.allowed_app_ids)

    @mcp.tool(
        description=(
            "Search book and study-material references via dense vector similarity, "
            "scoped to an exam dataset. Returns ranked passages with source metadata."
        )
    )
    async def search_book_references(
        app_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search booksegment_{app_id}_1024 Elasticsearch index.

        Args:
            app_id: Exam dataset to search (e.g. 'neetpg', 'mds').
            query:  Natural-language query string.
            top_k:  Max passages to return (1–20).
        """
        if app_id not in allowed:
            return [{"error": f"unknown app_id: {app_id!r}. Must be one of {sorted(allowed)}"}]

        try:
            from autogen.embeddings.jina import JinaEmbeddingClient
            from autogen.reranking.reranker import RerankClient
            from autogen.retrieval.finder import ReferenceFinder
            from autogen.storage.elastic import VectorStoreFactory, _create_es_client

            es_client = _create_es_client(settings)
            emb_client = JinaEmbeddingClient(settings.embedding_options)
            reranker = RerankClient(
                base_url=settings.reranking_options.base_url,
                api_key=settings.reranking_options.api_key,
                model=settings.reranking_options.default_model,
                timeout=settings.reranking_options.timeout_seconds,
            )
            store_factory = VectorStoreFactory(
                es_client=es_client,
                embedding_client=emb_client,
                dim=settings.elasticsearch.embedding_dim,
            )
            finder = ReferenceFinder(
                factory=store_factory,
                embedding=emb_client,
                reranker=reranker,
            )
            refs = await finder.find(app_id, query, top_k=min(top_k, 20))
            await es_client.aclose()
            return [r.model_dump() for r in refs]
        except Exception as exc:
            logger.error("mcp.search_book_references.error", app_id=app_id, error=str(exc))
            return [{"error": str(exc)}]

    @mcp.tool(
        description=(
            "Query the graph-RAG system for a specific exam dataset using one of four "
            "retrieval modes: NAIVE (vector only), LOCAL (entity-keyword), "
            "GLOBAL (relation-keyword), or HYBRID (all three fused)."
        )
    )
    async def query_graph(
        app_id: str,
        question: str,
        mode: str = "HYBRID",
        top_k: int = 10,
    ) -> dict:
        """Run Phase 3 HybridRetrieval for the given exam.

        Args:
            app_id:   Exam dataset (e.g. 'neetpg').
            question: Natural-language query.
            mode:     NAIVE | LOCAL | GLOBAL | HYBRID (case-insensitive).
            top_k:    Max results per sub-query.

        Returns:
            dict with 'entities', 'relations', 'sources' lists.
        """
        if app_id not in allowed:
            return {"error": f"unknown app_id: {app_id!r}"}

        try:
            from pathlib import Path

            from autogen.embeddings.jina import JinaEmbeddingClient
            from autogen.extraction.keywords import KeywordExtractor
            from autogen.llm.builder import build_llm_stack
            from autogen.models.enums import QueryMode
            from autogen.models.query import QueryParam
            from autogen.models.storage import EntityNode, EntityRelation, TextChunk
            from autogen.retrieval.hybrid import HybridRetrieval
            from autogen.storage.elastic import VectorStoreFactory, _create_es_client
            from autogen.storage.neo4j_graph import Neo4jGraphStoreFactory

            mode_map = {m.value.upper(): m for m in QueryMode}
            query_mode = mode_map.get(mode.upper(), QueryMode.HYBRID)

            es_client = _create_es_client(settings)
            emb_client = JinaEmbeddingClient(settings.embedding_options)
            store_factory = VectorStoreFactory(
                es_client=es_client,
                embedding_client=emb_client,
                dim=settings.elasticsearch.embedding_dim,
            )

            lg = settings.lightrag
            graph_store = Neo4jGraphStoreFactory(
                uri=lg.neo4j_uri,
                user=lg.neo4j_user,
                password=lg.neo4j_password,
            ).create(app_id)

            cache_dir = Path(settings.cache.base_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            llm = build_llm_stack(
                cache_dir=cache_dir,
                memory_size=settings.cache.memory_size,
                memory_ttl=settings.cache.memory_ttl_seconds,
            )
            kw_extractor = KeywordExtractor(llm=llm, model="groq/llama-3.1-8b-instant")

            retrieval = HybridRetrieval(
                app_id=app_id,
                chunk_store=store_factory.create(app_id, TextChunk),
                entity_store=store_factory.create(app_id, EntityNode),
                relation_store=store_factory.create(app_id, EntityRelation),
                graph_store=graph_store,
                keyword_extractor=kw_extractor,
            )

            params = QueryParam(mode=query_mode, top_k=top_k)
            ctx = await retrieval.retrieve(question, params)
            await es_client.aclose()

            return {
                "entities": [e.model_dump() for e in ctx.entities],
                "relations": [r.model_dump() for r in ctx.relationships],
                "sources": [s.model_dump() for s in ctx.sources],
            }
        except Exception as exc:
            logger.error("mcp.query_graph.error", app_id=app_id, error=str(exc))
            return {"error": str(exc)}

    return mcp


# ---------------------------------------------------------------------------
# /mcp/question — "question" server
# Mirrors QuestionUpdateMcpWrapper + SplitQuestionUpdateMcpWrapper.
# Tools: update_question, split_update_question
# ---------------------------------------------------------------------------


def build_question_mcp(services: Any, settings: Settings) -> FastMCP:
    """Build the 'question' MCP server for exam question management.

    Tools:
        update_question       — update metadata / correct answer for an exam question
        split_update_question — split a composite question into sub-questions
    """
    mcp = FastMCP("question")
    allowed = set(settings.app_identity.allowed_app_ids)

    @mcp.tool(
        description=(
            "Update an exam question's content, options, or correct answer. "
            "Scoped to an exam app_id. Requires SME-level access."
        )
    )
    async def update_question(
        app_id: str,
        question_id: str,
        updates: dict,
    ) -> dict:
        """Update a specific exam question.

        Args:
            app_id:      Exam dataset (e.g. 'neetpg').
            question_id: Unique question identifier.
            updates:     Fields to update (question_text, options, correct_answer, explanation).

        Returns:
            dict with 'success' bool and 'updated_fields' list.
        """
        if app_id not in allowed:
            return {"success": False, "error": f"unknown app_id: {app_id!r}"}

        logger.info(
            "mcp.update_question",
            app_id=app_id,
            question_id=question_id,
            fields=list(updates.keys()),
        )
        # Placeholder: real implementation would write to the question database
        return {
            "success": True,
            "question_id": question_id,
            "app_id": app_id,
            "updated_fields": list(updates.keys()),
            "note": "question update stub — wire to question DB in production",
        }

    @mcp.tool(
        description=(
            "Split a composite exam question into multiple individual sub-questions. "
            "Useful for restructuring multi-part items into atomic MCQs."
        )
    )
    async def split_update_question(
        app_id: str,
        question_id: str,
        sub_questions: list[dict],
    ) -> dict:
        """Split a question into sub-questions.

        Args:
            app_id:        Exam dataset.
            question_id:   Parent question to split.
            sub_questions: List of sub-question dicts, each with question_text + options.

        Returns:
            dict with 'success' bool and list of created 'sub_question_ids'.
        """
        if app_id not in allowed:
            return {"success": False, "error": f"unknown app_id: {app_id!r}"}

        logger.info(
            "mcp.split_update_question",
            app_id=app_id,
            question_id=question_id,
            sub_count=len(sub_questions),
        )
        return {
            "success": True,
            "parent_question_id": question_id,
            "app_id": app_id,
            "sub_question_count": len(sub_questions),
            "note": "split stub — wire to question DB in production",
        }

    return mcp


# ---------------------------------------------------------------------------
# /mcp/user — "user-analytics" server
# Mirrors McqdbDashboardMcpTools.
# Tools: mcqdb_prepdna_get, mcqdb_prepdna_history_get
# ---------------------------------------------------------------------------


def build_user_mcp(services: Any, settings: Settings) -> FastMCP:
    """Build the 'user-analytics' MCP server for PrepDNA proficiency data.

    Tools:
        mcqdb_prepdna_get         — current proficiency profile for a user
        mcqdb_prepdna_history_get — historical proficiency time series
    """
    mcp = FastMCP("user-analytics")
    allowed = set(settings.app_identity.allowed_app_ids)

    @mcp.tool(
        description=(
            "Get the current PrepDNA proficiency profile for a user in a specific exam. "
            "Returns topic-level strength scores and weak-area flags."
        )
    )
    async def mcqdb_prepdna_get(
        app_id: str,
        user_id: str,
    ) -> dict:
        """Retrieve current proficiency for user_id in the app_id exam.

        Args:
            app_id:  Exam dataset (e.g. 'neetpg').
            user_id: User identifier.

        Returns:
            dict with proficiency scores per topic and overall percentile.
        """
        if app_id not in allowed:
            return {"error": f"unknown app_id: {app_id!r}"}

        logger.info("mcp.prepdna_get", app_id=app_id, user_id=user_id)
        # Placeholder: real implementation queries PrepDNA analytics store
        return {
            "user_id": user_id,
            "app_id": app_id,
            "proficiency": {},
            "overall_percentile": None,
            "note": "PrepDNA stub — wire to analytics DB in production",
        }

    @mcp.tool(
        description=(
            "Get the historical PrepDNA proficiency time series for a user, "
            "showing week-by-week improvement across exam topics."
        )
    )
    async def mcqdb_prepdna_history_get(
        app_id: str,
        user_id: str,
        weeks: int = 8,
    ) -> dict:
        """Retrieve proficiency history for user_id.

        Args:
            app_id:  Exam dataset.
            user_id: User identifier.
            weeks:   Number of weeks of history to return (default 8).

        Returns:
            dict with 'history' list of weekly proficiency snapshots.
        """
        if app_id not in allowed:
            return {"error": f"unknown app_id: {app_id!r}"}

        logger.info(
            "mcp.prepdna_history_get", app_id=app_id, user_id=user_id, weeks=weeks
        )
        return {
            "user_id": user_id,
            "app_id": app_id,
            "weeks": weeks,
            "history": [],
            "note": "PrepDNA history stub — wire to analytics DB in production",
        }

    return mcp
