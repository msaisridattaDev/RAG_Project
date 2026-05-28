"""HybridRetrieval — four QueryMode paths for Phase 3 graph-aware retrieval.

Mirrors autogen.net's LightRag.cs query() method, dispatching on QueryMode:

  NAIVE  — vector search over TextChunks only (plain RAG).
  LOCAL  — local keywords → entity vector search → 1-hop graph expansion →
            source chunk lookup.
  GLOBAL — global keywords → relation vector search → endpoint entity fetch.
  HYBRID — LOCAL + GLOBAL merged via Reciprocal Rank Fusion.

Every path returns a CombinedContext (entities, relationships, sources) so the
QnA agent can assemble a single structured prompt regardless of mode.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from autogen.logging.setup import get_logger
from autogen.models.enums import QueryMode
from autogen.models.query import CombinedContext, QueryParam
from autogen.models.storage import EntityNode, EntityRelation, TextChunk
from autogen.reranking.elbow import find_elbow_index
from autogen.reranking.rrf import combine_results

if TYPE_CHECKING:
    from autogen.extraction.keywords import KeywordExtractor
    from autogen.protocols.graph import GraphStore
    from autogen.reranking.reranker import RerankClient
    from autogen.storage.elastic import ElasticVectorStore, VectorStoreFactory

logger = get_logger("autogen.retrieval.hybrid")

_RRF_K = 60


class HybridRetrieval:
    """Graph-aware retrieval for all four QueryMode paths.

    Usage::

        retrieval = HybridRetrieval(
            app_id="neetpg",
            chunk_store=chunk_store,
            entity_store=entity_store,
            relation_store=relation_store,
            graph_store=graph_store,
            keyword_extractor=keyword_extractor,
        )
        ctx = await retrieval.retrieve("What is aspirin?", QueryParam())
    """

    def __init__(
        self,
        *,
        app_id: str,
        chunk_store: ElasticVectorStore[TextChunk],
        entity_store: ElasticVectorStore[EntityNode],
        relation_store: ElasticVectorStore[EntityRelation],
        graph_store: GraphStore,
        keyword_extractor: KeywordExtractor,
        reranker: RerankClient | None = None,
    ) -> None:
        self._app_id = app_id
        self._chunks = chunk_store
        self._entities = entity_store
        self._relations = relation_store
        self._graph = graph_store
        self._kw = keyword_extractor
        self._reranker = reranker

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, params: QueryParam) -> CombinedContext:
        """Dispatch to the appropriate retrieval path and return CombinedContext."""
        mode = params.mode
        if mode == QueryMode.NAIVE:
            return await self._naive(query, params)
        if mode == QueryMode.LOCAL:
            return await self._local(query, params)
        if mode == QueryMode.GLOBAL:
            return await self._global(query, params)
        # HYBRID — 3-way parallel: LOCAL + GLOBAL + chunk-by-keyword
        local_ctx, global_ctx, keyword_ctx = await asyncio.gather(
            self._local(query, params),
            self._global(query, params),
            self._chunk_keyword(query, params),
        )
        return await self._merge_hybrid_three(query, local_ctx, global_ctx, keyword_ctx, params)

    # ------------------------------------------------------------------
    # NAIVE — plain vector search over chunks
    # ------------------------------------------------------------------

    async def _naive(self, query: str, params: QueryParam) -> CombinedContext:
        hits = await self._chunks.embedding_search(query, top_k=params.top_k)
        sources = [item for item, _score in hits]
        logger.debug("hybrid.naive", app_id=self._app_id, hits=len(sources))
        return CombinedContext(sources=sources, metadata={"mode": "Naive"})

    # ------------------------------------------------------------------
    # LOCAL — entity search → graph expansion → chunk lookup
    # ------------------------------------------------------------------

    async def _local(self, query: str, params: QueryParam) -> CombinedContext:
        local_kws, _ = await self._kw.extract(query)
        if not local_kws:
            local_kws = [query]

        # Over-fetch per keyword then apply elbow cutoff, then RRF
        overfetch = max(params.top_k * 3, 30)
        per_kw: list[list[tuple[str, float]]] = []
        for kw in local_kws[:6]:
            hits = await self._entities.embedding_search(kw, top_k=overfetch)
            if hits:
                scores = [s for _, s in hits]
                cutoff = find_elbow_index(scores) + 1
                hits = hits[:cutoff]
            per_kw.append([(item.id, score) for item, score in hits])

        # RRF fusion across keyword results
        fused = combine_results(per_kw, k=_RRF_K, top_n=params.top_k)
        entity_ids = [eid for eid, _ in fused]

        # Fetch full entity objects from graph
        entities = await self._graph.get_nodes(entity_ids)

        # 1-hop graph expansion — get incident edges, rank by strength
        relations = await self._graph.get_node_edges(entity_ids)
        relations = relations[: params.top_k]

        # Score chunks by how many 1-hop edges reference them, then fetch
        chunk_score: dict[str, int] = {}
        for ent in entities:
            for sid in ent.source_ids:
                chunk_score[sid] = chunk_score.get(sid, 0)
        for rel in relations:
            for sid in rel.source_ids:
                chunk_score[sid] = chunk_score.get(sid, 0) + 1

        # Sort by relation-count desc, then take top_k
        ranked_chunk_ids = sorted(
            chunk_score, key=lambda cid: chunk_score[cid], reverse=True
        )[: params.top_k]
        sources = await self._chunks.query_by_ids(ranked_chunk_ids)

        # Rerank with Qwen3 if available
        sources = await self._rerank(query, sources)

        logger.debug(
            "hybrid.local",
            app_id=self._app_id,
            keywords=len(local_kws),
            entities=len(entities),
            relations=len(relations),
            sources=len(sources),
        )
        return CombinedContext(
            entities=entities,
            relationships=relations,
            sources=sources,
            metadata={"mode": "Local", "keywords": local_kws},
        )

    # ------------------------------------------------------------------
    # GLOBAL — relation search → entity endpoint fetch
    # ------------------------------------------------------------------

    async def _global(self, query: str, params: QueryParam) -> CombinedContext:
        _, global_kws = await self._kw.extract(query)
        if not global_kws:
            global_kws = [query]

        # Over-fetch per keyword then apply elbow cutoff, then RRF
        overfetch = max(params.top_k * 3, 30)
        per_kw: list[list[tuple[str, float]]] = []
        for kw in global_kws[:4]:
            hits = await self._relations.embedding_search(kw, top_k=overfetch)
            if hits:
                scores = [s for _, s in hits]
                cutoff = find_elbow_index(scores) + 1
                hits = hits[:cutoff]
            per_kw.append([(item.id, score) for item, score in hits])

        fused = combine_results(per_kw, k=_RRF_K, top_n=params.top_k)
        rel_ids = [rid for rid, _ in fused]

        # Fetch full relation objects from graph
        relations = await self._graph.get_relations(rel_ids)

        # Fetch endpoint entities for context
        node_ids: list[str] = []
        seen_nids: set[str] = set()
        for rel in relations:
            for nid in (rel.source_id, rel.target_id):
                if nid and nid not in seen_nids:
                    node_ids.append(nid)
                    seen_nids.add(nid)

        entities = await self._graph.get_nodes(node_ids[: params.top_k])

        # Resolve chunks from relation source_ids
        chunk_ids: list[str] = []
        seen_cids: set[str] = set()
        for rel in relations:
            for sid in rel.source_ids:
                if sid not in seen_cids:
                    chunk_ids.append(sid)
                    seen_cids.add(sid)

        sources = await self._chunks.query_by_ids(chunk_ids[: params.top_k])

        # Rerank with Qwen3 if available
        sources = await self._rerank(query, sources)

        logger.debug(
            "hybrid.global",
            app_id=self._app_id,
            keywords=len(global_kws),
            relations=len(relations),
            entities=len(entities),
            sources=len(sources),
        )
        return CombinedContext(
            entities=entities,
            relationships=relations,
            sources=sources,
            metadata={"mode": "Global", "keywords": global_kws},
        )


    # ------------------------------------------------------------------
    # HYBRID path 3 — chunk-by-keyword (BM25-style keyword text search)
    # ------------------------------------------------------------------

    async def _chunk_keyword(self, query: str, params: QueryParam) -> CombinedContext:
        """Retrieve chunks whose ``keywords`` field overlaps with the query keywords.

        Uses the chunk store's keyword_search if available, otherwise falls back
        to an embedding search restricted to ``keyword_top_k`` results. The intent
        is to provide a lexical (not purely semantic) retrieval signal for HYBRID mode.
        """
        local_kws, global_kws = await self._kw.extract(query)
        all_kws = list(dict.fromkeys(local_kws + global_kws))  # dedup, preserve order
        if not all_kws:
            all_kws = [query]

        per_kw: list[list[tuple[str, float]]] = []
        for kw in all_kws[:4]:
            if hasattr(self._chunks, "keyword_search"):
                hits = await self._chunks.keyword_search(kw, top_k=params.keyword_top_k)
            else:
                hits = await self._chunks.embedding_search(kw, top_k=params.keyword_top_k)
            per_kw.append([(item.id, score) for item, score in hits])

        fused = combine_results(per_kw, k=_RRF_K, top_n=params.keyword_top_k)
        chunk_ids = [cid for cid, _ in fused]
        sources = await self._chunks.query_by_ids(chunk_ids)

        logger.debug(
            "hybrid.keyword",
            app_id=self._app_id,
            keywords=len(all_kws),
            sources=len(sources),
        )
        return CombinedContext(sources=sources, metadata={"mode": "Keyword", "keywords": all_kws})

    # ------------------------------------------------------------------
    # HYBRID 3-way merge + Qwen rerank
    # ------------------------------------------------------------------

    async def _merge_hybrid_three(
        self,
        query: str,
        local_ctx: CombinedContext,
        global_ctx: CombinedContext,
        keyword_ctx: CombinedContext,
        params: QueryParam,
    ) -> CombinedContext:
        """Merge LOCAL (30) + GLOBAL (30) + keyword (10) via RRF, then Qwen rerank."""

        # --- entities (local + global only) ---
        local_ents = [(e.id, 1.0) for e in local_ctx.entities]
        global_ents = [(e.id, 1.0) for e in global_ctx.entities]
        fused_ents = combine_results([local_ents, global_ents], k=_RRF_K, top_n=params.local_top_k)
        ent_id_order = {eid: rank for rank, (eid, _) in enumerate(fused_ents)}
        all_entities = {e.id: e for e in local_ctx.entities + global_ctx.entities}
        entities = sorted(
            [all_entities[eid] for eid in ent_id_order if eid in all_entities],
            key=lambda e: ent_id_order[e.id],
        )[: params.top_k]

        # --- relations (local + global only) ---
        local_rels = [(r.id, 1.0) for r in local_ctx.relationships]
        global_rels = [(r.id, 1.0) for r in global_ctx.relationships]
        fused_rels = combine_results([local_rels, global_rels], k=_RRF_K, top_n=params.global_top_k)
        rel_id_order = {rid: rank for rank, (rid, _) in enumerate(fused_rels)}
        all_rels = {r.id: r for r in local_ctx.relationships + global_ctx.relationships}
        relations = sorted(
            [all_rels[rid] for rid in rel_id_order if rid in all_rels],
            key=lambda r: rel_id_order[r.id],
        )[: params.top_k]

        # --- sources: 3-way RRF (local_top_k + global_top_k + keyword_top_k) ---
        local_srcs = [(c.id, 1.0) for c in local_ctx.sources]
        global_srcs = [(c.id, 1.0) for c in global_ctx.sources]
        keyword_srcs = [(c.id, 1.0) for c in keyword_ctx.sources]
        fused_srcs = combine_results(
            [local_srcs, global_srcs, keyword_srcs],
            k=_RRF_K,
            top_n=params.local_top_k + params.global_top_k + params.keyword_top_k,
        )
        src_id_order = {cid: rank for rank, (cid, _) in enumerate(fused_srcs)}
        all_srcs = {c.id: c for c in local_ctx.sources + global_ctx.sources + keyword_ctx.sources}
        sources = sorted(
            [all_srcs[cid] for cid in src_id_order if cid in all_srcs],
            key=lambda c: src_id_order[c.id],
        )

        # Qwen rerank over the merged candidate pool, final top_k
        sources = await self._rerank(query, sources)
        sources = sources[: params.top_k]

        logger.debug(
            "hybrid.merge_three",
            entities=len(entities),
            relations=len(relations),
            sources=len(sources),
        )
        return CombinedContext(
            entities=entities,
            relationships=relations,
            sources=sources,
            metadata={
                "mode": "Hybrid",
                "local_keywords": local_ctx.metadata.get("keywords", []),
                "global_keywords": global_ctx.metadata.get("keywords", []),
                "keyword_path_keywords": keyword_ctx.metadata.get("keywords", []),
            },
        )

    # ------------------------------------------------------------------
    # Reranking helper
    # ------------------------------------------------------------------

    async def _rerank(
        self, query: str, chunks: list[TextChunk]
    ) -> list[TextChunk]:
        """Rerank chunks with Qwen3-Reranker-4B if a reranker is configured.

        Gracefully falls back to the original order if reranking fails or
        no reranker is wired in (e.g., during tests or when the Qwen server
        is unavailable).
        """
        if not self._reranker or not chunks:
            return chunks

        texts = [c.content for c in chunks]
        try:
            hits = await self._reranker.rerank(query, texts, top_k=len(chunks))
            reranked = [chunks[h.index] for h in hits if h.index < len(chunks)]
            return reranked if reranked else chunks
        except Exception:
            logger.warning(
                "hybrid.rerank_failed — falling back to original order",
                exc_info=True,
            )
            return chunks


