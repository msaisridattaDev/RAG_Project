"""Embedding client protocol — mirrors autogen.net IEmbeddingClient.

Defines the interface for turning text into vectors.
Implementations: Jina (remote API), llama.cpp (self-hosted), OpenAI, etc.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Interface for embedding text into vectors.

    Implementations must handle batching, retries, and rate-limit safety.
    The embed method accepts a list of strings and returns a list of vectors
    in the same order as the input.
    """

    async def embed(
        self,
        texts: list[str],
        *,
        task: str = "retrieval.passage",
    ) -> list[list[float]]:
        """Embed a list of text strings into vectors.

        Args:
            texts: The text strings to embed.
            task: The embedding task type.
                - "retrieval.passage" for documents being indexed
                - "retrieval.query" for search queries

        Returns:
            A list of embedding vectors, one per input text, in input order.
            Each vector is a list of floats (typically 1024 dimensions).

        Raises:
            httpx.HTTPStatusError: If the provider returns a non-2xx status
                after all retries are exhausted.
            httpx.RequestError: If the request cannot be sent after retries.
        """
        ...
