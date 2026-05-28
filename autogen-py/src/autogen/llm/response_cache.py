"""Response cache — JSON serialization over EmbeddedKvCache + in-process memory.

Implements the ResponseCache protocol from autogen.protocols.llm.
Two-layer architecture:
  1. In-process cachetools.TTLCache (1 000 entries, ~1 µs hit)
  2. Embedded (aiosqlite) daily-rotated file (unlimited, ~50 µs hit)

LayeredResponseCache is the production implementation — it composes both layers
and populates the memory cache on any file hit.
"""

from __future__ import annotations

import json
from pathlib import Path

from cachetools import TTLCache
from structlog import get_logger

from autogen.llm.kv_cache import EmbeddedKvCache
from autogen.models.llm import LlmChunk, LlmUsage
from autogen.protocols.llm import ResponseCache

logger = get_logger(__name__)


class EmbeddedResponseCache:
    """File-backed ResponseCache using daily-rotated aiosqlite.

    Stores LlmChunk lists as JSON blobs under SHA256 keys.
    Implements the ResponseCache protocol so it can be swapped into
    CachingLlmClient or stacked inside LayeredResponseCache.
    """

    def __init__(self, kv: EmbeddedKvCache) -> None:
        self._kv = kv

    # ------------------------------------------------------------------
    # ResponseCache protocol
    # ------------------------------------------------------------------

    async def get(self, key: str) -> list[LlmChunk] | None:
        """Look up cached chunks for *key*. Returns None on miss or decode failure."""
        raw = await self._kv.get(key)
        if raw is None:
            return None
        try:
            return self._deserialize(raw)
        except Exception as exc:
            logger.debug("response_cache.decode_failed", key=key[:16], error=str(exc))
            return None

    async def set(self, key: str, chunks: list[LlmChunk]) -> None:
        """Store *chunks* as a JSON blob under *key*."""
        blob = self._serialize(chunks)
        await self._kv.set(key, blob)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(chunks: list[LlmChunk]) -> str:
        """Convert a list of LlmChunk to a compact JSON string."""
        payload = []
        for c in chunks:
            entry: dict[str, object] = {
                "delta": c.delta,
                "finish_reason": c.finish_reason,
                "is_cached": c.is_cached,
            }
            if c.usage:
                entry["usage"] = {
                    "prompt_tokens": c.usage.prompt_tokens,
                    "completion_tokens": c.usage.completion_tokens,
                    "total_tokens": c.usage.total_tokens,
                    "total_cost": c.usage.total_cost,
                    "is_cached": c.usage.is_cached,
                    "model": c.usage.model,
                }
            payload.append(entry)
        return json.dumps(payload, separators=(",", ":"))

    @staticmethod
    def _deserialize(raw: str) -> list[LlmChunk]:
        """Reconstruct a list of LlmChunk from a JSON blob."""
        payload = json.loads(raw)
        out: list[LlmChunk] = []
        for entry in payload:
            usage = None
            if "usage" in entry:
                u = entry["usage"]
                usage = LlmUsage(
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    total_tokens=u.get("total_tokens", 0),
                    total_cost=u.get("total_cost", 0.0),
                    is_cached=u.get("is_cached", False),
                    model=u.get("model"),
                )
            out.append(
                LlmChunk(
                    delta=entry.get("delta", ""),
                    finish_reason=entry.get("finish_reason"),
                    usage=usage,
                    is_cached=entry.get("is_cached", False),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Layered (in-process + embedded) cache
# ---------------------------------------------------------------------------


class LayeredResponseCache:
    """Two-layer ResponseCache: in-process TTLCache + EmbeddedResponseCache.

    On read: check memory first; if miss, check embedded; if embedded hit,
    populate memory for next time.

    On write: write to both layers.

    This is NOT a ResponseCache protocol implementor — it wraps two
    ResponseCache instances and provides the same get/set API.
    """

    def __init__(
        self,
        memory: TTLCache,
        embedded: EmbeddedResponseCache,
    ) -> None:
        self._memory = memory
        self._embedded = embedded

    async def get(self, key: str) -> list[LlmChunk] | None:
        # 1. Memory layer
        cached = self._memory.get(key)
        if cached is not None:
            return list(cached)  # Return a copy to avoid mutation

        # 2. Embedded layer
        from_embedded = await self._embedded.get(key)
        if from_embedded is not None:
            # Populate memory for next read
            self._memory[key] = from_embedded
            return list(from_embedded)

        return None

    async def set(self, key: str, chunks: list[LlmChunk]) -> None:
        # Write both layers (embedded first — if that fails, memory isn't stale)
        await self._embedded.set(key, chunks)
        self._memory[key] = chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_response_cache(cache_dir: Path, memory_maxsize: int = 1000) -> LayeredResponseCache:
    """Build the production-ready two-layer cache.

    Args:
        cache_dir: Directory for daily SQLite files.
        memory_maxsize: Max entries in the in-process TTLCache.
    """
    kv = EmbeddedKvCache(cache_dir)
    embedded = EmbeddedResponseCache(kv)
    memory = TTLCache(maxsize=memory_maxsize, ttl=3600)  # 1-hour TTL in memory
    return LayeredResponseCache(memory=memory, embedded=embedded)