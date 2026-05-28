"""CachingLlmClient — decorator that caches streaming responses.

Mirrors autogen.net's CachingClient.cs (wrapping an inner ILlmClient).

On cache hit, replays cached chunks with 20ms pacing so the user experience
looks identical to a fresh provider response. The is_cached flag on every
chunk is set to True, enabling the usage tracker (outer decorator) to route
the cost to _cached instead of _real.

Cache misses are forwarded to the inner client; chunks are accumulated and
written to the cache only after the stream completes (atomicity: no partial
entries that would replay truncated answers).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from structlog import get_logger

from autogen.llm.cache_key import cache_key as compute_cache_key
from autogen.llm.response_cache import LayeredResponseCache
from autogen.models.llm import LlmChunk, LlmMessage
from autogen.protocols.llm import LlmClient

logger = get_logger(__name__)

# The 20ms replay delay preserved from CachingMiddleware.cs:50.
# Keeps cached responses visually indistinguishable from live streaming.
CACHE_REPLAY_DELAY = 0.02


class CachingLlmClient:
    """Decorator: caches LlmClient.stream() calls at the SHA256 level.

    Wraps an inner LlmClient (typically LiteLlmClient). The outer
    UsageTrackingLlmClient wraps *this* so tracking sees cached calls.

    Usage::

        raw = LiteLlmClient()
        cache = LayeredResponseCache(...)
        cached = CachingLlmClient(inner=raw, cache=cache)
        async for chunk in cached.stream(msgs, "groq/llama-3.3-70b"):
            ...
    """

    def __init__(
        self,
        inner: LlmClient,
        cache: LayeredResponseCache,
    ) -> None:
        self._inner = inner
        self._cache = cache

    # ------------------------------------------------------------------
    # LlmClient protocol
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[LlmMessage],
        model: str,
        **kwargs: object,
    ) -> Any:  # AsyncIterator[LlmChunk]
        return _CachedStreamIterator(
            caching_client=self,
            messages=messages,
            model=model,
            kwargs=kwargs,
        )

    async def complete(
        self,
        messages: list[LlmMessage],
        model: str,
        **kwargs: object,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.stream(messages, model, **kwargs):
            if chunk.delta:
                parts.append(chunk.delta)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Internal async iterator — the core of caching logic
# ---------------------------------------------------------------------------


class _CachedStreamIterator:
    """Async iterator returned by CachingLlmClient.stream().

    Path A (cache hit):  replay cached chunks with 20ms pacing.
    Path B (cache miss): forward to inner, accumulate, write cache on completion.
    """

    def __init__(
        self,
        caching_client: CachingLlmClient,
        messages: list[LlmMessage],
        model: str,
        kwargs: dict[str, object],
    ) -> None:
        self._inner = caching_client._inner
        self._cache = caching_client._cache
        self._messages = messages
        self._model = model
        self._kwargs = kwargs
        self._hit = False
        self._accumulated: list[LlmChunk] = []

        # Extract parameters that affect the cache key
        temperature = kwargs.pop("temperature", 0.0)
        response_format = kwargs.pop("response_format", None)
        self._cache_key = compute_cache_key(
            messages, model, temperature, response_format
        )

    def __aiter__(self) -> "_CachedStreamIterator":
        return self

    async def _setup(self) -> None:
        """Lazily check cache and initialise the appropriate path."""
        cached = await self._cache.get(self._cache_key)
        if cached is not None:
            logger.debug("cache.hit", key=self._cache_key[:16])
            self._hit = True
            self._cached_chunks = cached
            self._cached_index = 0
        else:
            logger.debug("cache.miss", key=self._cache_key[:16])
            self._hit = False
            self._inner_stream = self._inner.stream(
                self._messages, self._model, **self._kwargs
            )
            self._inner_iter = None

    async def __anext__(self) -> LlmChunk:
        if not hasattr(self, "_cached_chunks") and not hasattr(self, "_inner_stream"):
            await self._setup()
        if self._hit:
            return await self._replay_next()
        else:
            return await self._forward_next()

    # ------------------------------------------------------------------
    # Cache hit path
    # ------------------------------------------------------------------

    async def _replay_next(self) -> LlmChunk:
        if self._cached_index >= len(self._cached_chunks):
            raise StopAsyncIteration

        chunk = self._cached_chunks[self._cached_index]
        self._cached_index += 1

        # Set is_cached on the chunk and its usage
        usage = chunk.usage
        if usage is not None:
            usage = usage.model_copy(update={"is_cached": True})

        replayed = LlmChunk(
            delta=chunk.delta,
            finish_reason=chunk.finish_reason,
            usage=usage,
            is_cached=True,
        )

        # 20ms replay delay (mirrors CachingMiddleware.cs:50)
        if self._cached_index < len(self._cached_chunks):
            await asyncio.sleep(CACHE_REPLAY_DELAY)

        return replayed

    # ------------------------------------------------------------------
    # Cache miss path
    # ------------------------------------------------------------------

    async def _forward_next(self) -> LlmChunk:
        # Lazy init the inner async iterator
        if self._inner_iter is None:
            self._inner_iter = self._inner_stream.__aiter__()

        try:
            chunk = await self._inner_iter.__anext__()
        except StopAsyncIteration:
            # End of inner stream — write accumulated chunks to cache
            if self._accumulated and not self._has_tool_calls():
                await self._cache.set(self._cache_key, self._accumulated)
                logger.debug("cache.stored", key=self._cache_key[:16])
            raise

        self._accumulated.append(chunk)
        return chunk

    def _has_tool_calls(self) -> bool:
        """Check if any accumulated chunk contains a tool (function) call.

        If the response involves tool calls, the same messages could
        legitimately produce different answers next time (tool result
        isn't in the cache key). Skip caching such responses.
        """
        for chunk in self._accumulated:
            # We don't store tool calls in LlmChunk explicitly, so we check
            # the finish_reason — "tool_calls" means tool calls were present.
            if chunk.finish_reason == "tool_calls":
                return True
        return False