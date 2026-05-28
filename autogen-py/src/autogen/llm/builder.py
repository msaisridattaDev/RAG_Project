"""LLM stack builder — wires LiteLLM + Cache + Usage into a composable client.

The build function constructs the decorator stack:
    UsageTracking → Caching → LiteLLM

This is the ONE function the rest of the system (DI container, QnAAgent)
calls to obtain an LLM client. Every layer above this builder gets streaming,
caching, usage tracking, and cost attribution without knowing any of them exist.

Mirrors autogen.net's DI registration where the HttpMessageHandler + CachingClient
chain is built (Program.cs:136-160 conceptually).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from autogen.llm.caching_client import CachingLlmClient
from autogen.llm.lite_llm import LiteLlmClient
from autogen.llm.response_cache import LayeredResponseCache
from autogen.llm.usage import InMemoryUsageCollector, UsageTrackingLlmClient
from autogen.protocols.llm import LlmClient


@lru_cache(maxsize=1)
def _singleton_collector() -> InMemoryUsageCollector:
    """Return the process-wide singleton UsageCollector."""
    return InMemoryUsageCollector()


def build_llm_stack(
    cache_dir: Path,
    memory_size: int = 1000,
    memory_ttl: int = 3600,
) -> LlmClient:
    """Build the fully decorated LLM client stack.

    Stack order (outermost first):
      UsageTrackingLlmClient  ← records cost to _total/_cached/_real
        └─ CachingLlmClient   ← SHA256 exact-match cache with 20ms replay
             └─ LiteLlmClient  ← raw LiteLLM adapter

    Args:
        cache_dir: Directory for daily cache files (e.g. ./cache/OpenLM).
        memory_size: Entries in the in-process TTLCache.
        memory_ttl: TTL in seconds for the in-process cache.

    Returns:
        A LlmClient that is fully decorated and ready for injection.
    """
    from autogen.llm.response_cache import create_response_cache

    # Innermost: raw LiteLLM adapter
    raw = LiteLlmClient()

    # Middle: caching decorator
    response_cache = create_response_cache(
        cache_dir=cache_dir,
        memory_maxsize=memory_size,
    )
    cached = CachingLlmClient(inner=raw, cache=response_cache)

    # Outermost: usage tracking
    collector = _singleton_collector()
    tracked = UsageTrackingLlmClient(inner=cached, collector=collector)

    return tracked


def get_usage_collector() -> InMemoryUsageCollector:
    """Return the singleton UsageCollector (for querying from endpoints)."""
    return _singleton_collector()