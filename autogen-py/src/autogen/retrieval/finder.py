"""ReferenceFinder — the vertical-slice integration of Days 3–5.

Composes embedding, vector search, elbow cutoff, reranking, and token
truncation into a single end-to-end retrieval pipeline, parameterized
by app_id at call time so one finder serves all exam datasets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tiktoken

from autogen.models.reference import Reference
from autogen.reranking.elbow import find_elbow_index

if TYPE_CHECKING:
    from autogen.protocols.embedding import EmbeddingClient
    from autogen.reranking.reranker import RerankClient
    from autogen.storage.elastic import ElasticVectorStore, VectorStoreFactory

logger = logging.getLogger(__name__)


class ReferenceFinder:
    """End-to-end retrieval pipeline — mirrors autogen.net ReferenceFinder.

    Composes Days 3–5 into one class. Parameterized by app_id at call time
    so one singleton instance serves all exam datasets.

    Pipeline:
        1. Resolve per-app store via factory
        2. Embed query
        3. Search index at top_k * 3 (over-fetch)
        4. Apply elbow cutoff
        5. Rerank survivors (with graceful fallback)
        6. Token-truncate to max_tokens
        7. Return list[Reference]
    """

    def __init__(
        self,
        factory: VectorStoreFactory,
        embedding: EmbeddingClient,
        reranker: RerankClient,
        tiktoken_model: str = "cl100k_base",
        overfetch_multiplier: int = 3,
    ) -> None:
        """Initialize the ReferenceFinder.

        Args:
            factory: VectorStoreFactory for resolving per-(type, app_id) stores.
            embedding: Embedding client (JinaEmbeddingClient or compatible).
            reranker: Rerank client (Qwen3-Reranker-4B or compatible).
            tiktoken_model: Tokenizer model name for truncation (default cl100k_base).
            overfetch_multiplier: How many times top_k to fetch from vector search.
        """
        self._factory = factory
        self._embedding = embedding
        self._reranker = reranker
        self._tiktoken_model = tiktoken_model
        self._overfetch_multiplier = overfetch_multiplier

    async def find(
        self,
        app_id: str,
        query: str,
        top_k: int = 10,
        max_tokens: int = 4000,
        store_type: type | None = None,
    ) -> list[Reference]:
        """Retrieve references for a query, scoped to an app_id.

        Args:
            app_id: The exam/dataset tenant ID (e.g., "neetpg", "mds").
            query: The natural language query string.
            top_k: Desired number of final results.
            max_tokens: Maximum total tokens across all returned references.
            store_type: The Pydantic model type for the vector store.
                       Defaults to TextChunk if not specified.

        Returns:
            A list of Reference objects, sorted by relevance descending.
        """
        # Lazy import to avoid circular dependency at module level
        from autogen.models.storage import TextChunk

        model_type = store_type or TextChunk

        # Step 1: Resolve per-app store via factory
        store: ElasticVectorStore = self._factory.create(app_id, model_type)

        # Step 2-3: Embed query and search at over-fetched top_k
        over_fetched_k = top_k * self._overfetch_multiplier
        candidates = await store.embedding_search(query, top_k=over_fetched_k)

        if not candidates:
            logger.info("finder.empty app_id=%s query=%s", app_id, query[:80])
            return []

        # Step 4: Apply elbow cutoff
        scores = [score for _, score in candidates]
        elbow_idx = find_elbow_index(scores)
        elbow_candidates = candidates[: elbow_idx + 1]

        if not elbow_candidates:
            logger.warning("finder.elbow_empty app_id=%s query=%s", app_id, query[:80])
            elbow_candidates = candidates[:top_k]

        # Step 5: Rerank survivors (with graceful fallback)
        candidate_texts = [item.content for item, _ in elbow_candidates]
        reranked: list[tuple] = await self._rerank_or_fallback(
            query, candidate_texts, elbow_candidates, top_k
        )

        # Step 6: Token-truncate to max_tokens
        enc = tiktoken.get_encoding(self._tiktoken_model)
        budget = max_tokens
        references: list[Reference] = []

        for idx, (chunk, score) in enumerate(reranked):
            toks = len(enc.encode(chunk.content))
            if toks > budget:
                continue  # skip chunks that don't fit
            budget -= toks

            ref = Reference(
                id=chunk.id if hasattr(chunk, "id") else f"result-{idx}",
                content=chunk.content,
                score=float(score),
                metadata={"app_id": app_id, "index": idx},
            )
            references.append(ref)

        logger.info(
            "finder.ok app_id=%s query=%s candidates=%d elbow=%d final=%d",
            app_id,
            query[:80],
            len(candidates),
            len(elbow_candidates),
            len(references),
        )
        return references

    async def _rerank_or_fallback(
        self,
        query: str,
        candidate_texts: list[str],
        elbow_candidates: list[tuple],
        top_k: int,
    ) -> list[tuple]:
        """Attempt reranking, fall back to vector order on failure.

        Returns a list of (item, score) tuples. On success, score is the
        reranker's relevance_score; on fallback, the original vector score.

        Mirrors QueryOperations.cs:856-862 graceful degradation pattern.
        """
        try:
            reranked = await self._reranker.rerank(query, candidate_texts, top_k=top_k)
            if reranked:
                return [
                    (elbow_candidates[h.index][0], h.relevance_score)
                    for h in reranked
                ]
        except Exception:
            logger.exception(
                "reranker.failed fallback=vector_order",
            )

        # Fallback: return in vector-search order, preserving vector scores
        return list(elbow_candidates[:top_k])
