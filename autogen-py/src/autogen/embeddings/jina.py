"""Jina Embeddings Client — mirrors autogen.net JinaEmbeddingClient.

Turns text into 1024-dimensional vectors via the Jina Embeddings API.
Implements the EmbeddingClient protocol with:
    - Batching at 32 inputs per API call
    - Concurrency cap at 4 simultaneous requests
    - Retry on 5xx with exponential backoff (via Tenacity)
    - Configurable via Settings.embedding_options
    - Secure logging: never logs API keys, request bodies, or response content
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from autogen.config.settings import EmbeddingSettings
from autogen.logging.setup import get_logger

logger = get_logger("autogen.embeddings.jina")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
"""Maximum number of texts per API call. Jina API has per-request size limits;
32 is a safe value across all tiers."""

MAX_CONCURRENCY = 4
"""Maximum number of concurrent API requests. Prevents provider rate-limit
hits and local resource exhaustion."""

RETRY_ATTEMPTS = 3
"""Number of retry attempts on 5xx / network errors."""

RETRY_MIN_WAIT = 1  # seconds
RETRY_MAX_WAIT = 10  # seconds

# ---------------------------------------------------------------------------
# Retry helpers — only retry on 5xx and network errors, never on 4xx
# ---------------------------------------------------------------------------


def _should_retry_on_status(response: httpx.Response) -> bool:
    """Return True only for 5xx status codes (server errors).

    4xx errors (bad request, auth failure, etc.) should NOT be retried
    because the client sent something wrong — retrying won't fix it.
    """
    return 500 <= response.status_code < 600


def _is_retryable_exception(exception: BaseException) -> bool:
    """Return True if the exception should trigger a retry.

    Only retry on 5xx HTTP errors (server-side problems that may be transient).
    Never retry on 4xx errors (client-side problems — bad input, bad auth, etc.).
    """
    if isinstance(exception, httpx.HTTPStatusError):
        return _should_retry_on_status(exception.response)
    # Network errors (ConnectionError, ReadTimeout, etc.) are also retryable
    return bool(isinstance(exception, httpx.RequestError))


_retry_decorator = retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception(_is_retryable_exception),
    reraise=True,
)

# ---------------------------------------------------------------------------
# JinaEmbeddingClient
# ---------------------------------------------------------------------------


class JinaEmbeddingClient:
    """Embedding client backed by the Jina Embeddings API.

    Implements the EmbeddingClient protocol. Configuration is read from
    Settings.embedding_options, which mirrors appsettings.json's
    EmbeddingOptions section field-for-field.

    Usage:
        client = JinaEmbeddingClient(settings.embedding_options)
        vectors = await client.embed(["text1", "text2", ...])
    """

    def __init__(
        self,
        options: EmbeddingSettings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Jina embedding client.

        Args:
            options: Embedding provider settings (base_url, model, api_key).
            http_client: Optional pre-configured httpx AsyncClient.
                If not provided, a default client is created.
        """
        self._options = options
        self._http_client = http_client or httpx.AsyncClient(timeout=60.0)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        # Validate that we have an API key
        if not options.api_key:
            logger.warning("jina.api_key.missing")

    async def embed(
        self,
        texts: list[str],
        *,
        task: str = "retrieval.passage",
    ) -> list[list[float]]:
        """Embed a list of text strings into vectors.

        Splits input into batches of 32, sends up to 4 concurrent requests,
        retries on 5xx with exponential backoff.

        Args:
            texts: The text strings to embed.
            task: Embedding task type.
                - "retrieval.passage" for documents being indexed (default)
                - "retrieval.query" for search queries

        Returns:
            A list of embedding vectors, one per input text, in input order.

        Raises:
            httpx.HTTPStatusError: If the provider returns a non-2xx status
                after all retries are exhausted.
            httpx.RequestError: If the request cannot be sent after retries.
        """
        if not texts:
            return []

        # Split into batches of BATCH_SIZE
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        logger.info(
            "jina.embed.start",
            total_texts=len(texts),
            batch_count=len(batches),
            batch_size=BATCH_SIZE,
            task=task,
        )

        # Wrap each batch in a semaphore-guarded coroutine
        async def _guarded_batch(batch: list[str]) -> list[list[float]]:
            async with self._semaphore:
                return await self._embed_batch(batch, task=task)

        # Launch all batches concurrently (capped by semaphore)
        results: list[list[list[float]]] = await asyncio.gather(
            *[_guarded_batch(b) for b in batches],
        )

        # Flatten: [batch_vectors_1, batch_vectors_2, ...] -> [v1, v2, ...]
        flattened: list[list[float]] = [v for batch in results for v in batch]

        logger.info(
            "jina.embed.complete",
            total_texts=len(texts),
            vectors_returned=len(flattened),
        )

        return flattened

    # ------------------------------------------------------------------
    # Internal: single batch embedding with retry
    # ------------------------------------------------------------------

    @_retry_decorator
    async def _embed_batch(
        self,
        batch: list[str],
        *,
        task: str = "retrieval.passage",
    ) -> list[list[float]]:
        """Embed a single batch of texts via the Jina API.

        This method is decorated with Tenacity retry logic:
            - Retries on httpx.HTTPStatusError (5xx only — see _should_retry_on_status)
            - Exponential backoff: 1s, 2s, 4s, ... capped at 10s
            - Up to 3 attempts
            - Reraises on final failure

        Args:
            batch: A list of texts to embed (max BATCH_SIZE items).
            task: The embedding task type.

        Returns:
            A list of embedding vectors for this batch.

        Raises:
            httpx.HTTPStatusError: If all retry attempts fail with 5xx.
            httpx.RequestError: If network errors persist after retries.
        """
        start_time = time.monotonic()

        url = f"{self._options.base_url.rstrip('/')}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._options.api_key:
            headers["Authorization"] = f"Bearer {self._options.api_key}"
        body: dict[str, Any] = {
            "model": self._options.default_model,
            "input": batch,
            "task": task,
        }

        response = await self._http_client.post(url, headers=headers, json=body)

        # Check for 4xx — these should NOT be retried
        if response.status_code < 500 and response.status_code >= 400:
            logger.error(
                "jina.embed.client_error",
                status_code=response.status_code,
                batch_size=len(batch),
                task=task,
            )
            response.raise_for_status()  # Will raise HTTPStatusError (not retried)

        # Raise for 5xx — the retry decorator will catch this
        if _should_retry_on_status(response):
            logger.warning(
                "jina.embed.server_error",
                status_code=response.status_code,
                batch_size=len(batch),
                task=task,
            )
            response.raise_for_status()  # Will raise HTTPStatusError (retried)

        # Parse successful response
        response.raise_for_status()
        data = response.json()

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Extract embeddings from response
        # Response format: {"data": [{"embedding": [...], "index": 0}, ...]}
        embeddings: list[list[float]] = [row["embedding"] for row in data["data"]]

        # Log timing and counts — NEVER log API keys, request body, or response content
        logger.info(
            "jina.embed.ok",
            batch_size=len(batch),
            elapsed_ms=elapsed_ms,
            task=task,
            model=self._options.default_model,
            # Token count if available in response
            token_count=data.get("usage", {}).get("total_tokens"),
        )

        return embeddings

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http_client.aclose()
