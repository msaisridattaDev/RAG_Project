"""Phase 2 — LLM Orchestration & Caching.

Public surface:
  - LlmClient (protocol) — what every decorator implements
  - LlmMessage, LlmChunk, LlmUsage — wire types
  - build_llm_stack() — the ONE function to construct the full decorator stack
  - TierModelRouter, create_router() — model selection by tier + role
  - ModelsCatalog, load_models_catalog() — hot-swappable model catalog
  - get_usage_collector() — access the process-wide usage ledger

Architecture (decorator stack, outermost first):
  UsageTrackingLlmClient → CachingLlmClient → LiteLlmClient
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Protocol & wire types — used by every layer
# ---------------------------------------------------------------------------
from autogen.protocols.llm import LlmClient, LlmChunk, LlmMessage, LlmUsage

# ---------------------------------------------------------------------------
# Core components
# ---------------------------------------------------------------------------
from autogen.llm.lite_llm import LiteLlmClient
from autogen.llm.cache_key import cache_key
from autogen.llm.caching_client import CachingLlmClient
from autogen.llm.usage import (
    InMemoryUsageCollector,
    UsageBucket,
    UsageTrackingLlmClient,
    SESSION_KEY_VAR,
)
from autogen.llm.catalog import (
    ModelsCatalog,
    ModelEntry,
    ProviderConfig,
    TierModelRouter,
    load_models_catalog,
    create_router,
    MODEL_ROLES,
)
from autogen.llm.builder import build_llm_stack, get_usage_collector

# ---------------------------------------------------------------------------
# What the rest of the system imports
# ---------------------------------------------------------------------------
__all__ = [
    # Protocol & types
    "LlmClient",
    "LlmMessage",
    "LlmChunk",
    "LlmUsage",
    # Stack builder (primary entry point)
    "build_llm_stack",
    "get_usage_collector",
    # Individual decorators (testing / advanced use)
    "LiteLlmClient",
    "CachingLlmClient",
    "UsageTrackingLlmClient",
    # Cache key
    "cache_key",
    # Usage
    "InMemoryUsageCollector",
    "UsageBucket",
    "SESSION_KEY_VAR",
    # Model catalog & router
    "ModelsCatalog",
    "ModelEntry",
    "ProviderConfig",
    "TierModelRouter",
    "load_models_catalog",
    "create_router",
    "MODEL_ROLES",
]