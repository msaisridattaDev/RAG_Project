"""Reranker protocol — mirrors autogen.net IRerankClient.

The concrete ``RerankClient`` in ``autogen.reranking.reranker`` is structurally
substitutable for this Protocol. Phase 1 Day 6's ``ReferenceFinder`` and
Phase 3 Day 16's hybrid query path type their dependency against this
interface so a Jina-hosted or self-hosted Qwen reranker can be swapped
without changing call sites.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from autogen.reranking.reranker import RerankHit


@runtime_checkable
class RerankClientProtocol(Protocol):
    """Cross-encoder reranker — mirrors autogen.net IRerankClient.

    Implementations: ``autogen.reranking.reranker.RerankClient`` (Qwen3 /
    Jina-reranker via /rerank wire format). The interface is intentionally
    minimal — Phase 1 caller is responsible for the graceful-fallback policy.
    """

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankHit]:
        """Rerank documents by relevance to the query.

        Returns ``RerankHit`` rows sorted by ``relevance_score`` descending.
        Implementations should NOT retry internally — the caller decides
        whether to fall back to vector order on failure.
        """
        ...
