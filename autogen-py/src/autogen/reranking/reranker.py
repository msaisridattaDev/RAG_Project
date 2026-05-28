"""RerankClient — cross-encoder reranker for two-stage retrieval.

POSTs to a /rerank endpoint (OpenAI-compatible wire format) that works against
either a self-hosted llama.cpp Qwen3-Reranker-4B server or the hosted Jina API.

Default model: Qwen/Qwen3-Reranker-4B (matching ElasticVectorDb.cs:29).

Usage:
    client = RerankClient(
        base_url="http://home.bhakars.com:8077",
        api_key=None,  # self-hosted, no key needed
        model="Qwen/Qwen3-Reranker-4B",
    )
    hits = await client.rerank(
        query="what causes inflammation?",
        documents=["doc1 text...", "doc2 text..."],
        top_k=5,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("autogen.reranking.reranker")


# ---------------------------------------------------------------------------
# RerankHit — one reranked result
# ---------------------------------------------------------------------------


@dataclass
class RerankHit:
    """A single reranked result from the cross-encoder.

    Attributes:
        index: Original position in the input documents list.
        relevance_score: Relevance score (higher is better).
        document: The document text echoed back.
    """

    index: int
    relevance_score: float
    document: str


# ---------------------------------------------------------------------------
# RerankClient
# ---------------------------------------------------------------------------


class RerankClient:
    """Cross-encoder reranker client.

    POSTs to a /rerank endpoint with an OpenAI-compatible request body:
        {
            "model": "...",
            "query": "...",
            "documents": ["...", "..."],
            "top_n": 5
        }

    Works against:
        - Self-hosted llama.cpp Qwen3-Reranker-4B (default)
        - Hosted Jina /rerank API (if configured)
    """

    def __init__(
        self,
        base_url: str = "http://home.bhakars.com:8077",
        api_key: str | None = None,
        model: str = "Qwen/Qwen3-Reranker-4B",
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the reranker client.

        Args:
            base_url: Base URL of the reranker endpoint.
                Defaults to the llama.cpp server address from the .NET source.
            api_key: Optional API key for hosted providers (e.g., Jina).
                Self-hosted Qwen typically doesn't require one.
            model: The reranker model name.
                Default: "Qwen/Qwen3-Reranker-4B" (matching ElasticVectorDb.cs:29).
            http_client: Optional pre-configured httpx.AsyncClient.
                If not provided, one is created.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankHit]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query string.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all.

        Returns:
            A list of RerankHit sorted by relevance_score descending.
            Empty list if documents is empty.

        Raises:
            httpx.HTTPStatusError: If the reranker returns a non-2xx status.
            httpx.RequestError: If the request fails (network error, timeout).
        """
        if not documents:
            return []

        url = f"{self._base_url}/rerank"

        body: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if top_k is not None:
            body["top_n"] = top_k

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        logger.debug(
            "reranker.request",
            extra={
                "url": url,
                "model": self._model,
                "doc_count": len(documents),
                "top_k": top_k,
            },
        )

        response = await self._client.post(url, json=body, headers=headers)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        hits = [
            RerankHit(
                index=item.get("index", i),
                relevance_score=float(item.get("relevance_score", 0.0)),
                document=item.get(
                    "document",
                    documents[item.get("index", i)]
                    if item.get("index", i) < len(documents)
                    else "",
                ),
            )
            for i, item in enumerate(results)
        ]

        # Sort by relevance_score descending (API should already do this,
        # but be defensive)
        hits.sort(key=lambda h: h.relevance_score, reverse=True)

        logger.info(
            "reranker.ok",
            extra={
                "url": url,
                "model": self._model,
                "input_count": len(documents),
                "output_count": len(hits),
                "top_score": hits[0].relevance_score if hits else None,
            },
        )

        return hits

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
