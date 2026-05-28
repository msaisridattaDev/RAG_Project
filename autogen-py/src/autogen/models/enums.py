"""Enumerations — mirrors autogen.net enum types.

All enum values used across the system are defined here to ensure
a single source of truth for every vocabulary term.
"""

from __future__ import annotations

from enum import StrEnum


class Tier(StrEnum):
    """Usage tier — determines rate limits, feature access, and storage quotas.

    Mirrors autogen.net Tier enum (Program.cs:172-284 — four tiers exactly).
    """

    FREE = "Free"
    TESTING = "Testing"
    REGULAR = "Regular"
    PREMIUM = "Premium"


# Plan name → existing enum. Re-exported so callers can import either symbol.
UserTier = Tier


class QueryMode(StrEnum):
    """Retrieval mode — drives which Phase 3 query path runs.

    Mirrors autogen.net QueryMode. LOCAL = entity-keyword path,
    GLOBAL = relationship-keyword path, HYBRID = both fused via RRF,
    NAIVE = plain vector RAG over chunks.
    """

    LOCAL = "Local"
    GLOBAL = "Global"
    HYBRID = "Hybrid"
    NAIVE = "Naive"


class AgentStatus(StrEnum):
    """Lifecycle status of a QnA agent.

    Mirrors autogen.net AgentStatus enum.
    """

    IDLE = "Idle"
    PROCESSING = "Processing"
    WAITING_FOR_INPUT = "WaitingForInput"
    COMPLETED = "Completed"
    ERROR = "Error"
    CANCELLED = "Cancelled"


class StoreKind(StrEnum):
    """Type of storage substrate.

    Mirrors autogen.net StoreKind enum.
    """

    VECTOR = "Vector"
    CACHE = "Cache"
    GRAPH = "Graph"
