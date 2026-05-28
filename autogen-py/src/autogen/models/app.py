"""Application-level configuration models — mirrors autogen.net AppConfig.

Defines the configuration shape for an application instance within
the multi-tenant system.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from autogen.models.base import AppId
from autogen.models.enums import Tier


class AppConfig(BaseModel):
    """Application configuration — mirrors autogen.net AppConfig.

    Describes the capabilities and limits of a specific app instance.
    """

    app_id: AppId
    tier: Tier = Tier.FREE
    max_concurrent_agents: int = Field(default=5, ge=1, le=100)
    max_documents_per_query: int = Field(default=10, ge=1, le=500)
    enable_graph_rag: bool = False
    enable_cache: bool = True
    rate_limit_rpm: int = Field(default=60, ge=1, description="Requests per minute")
