"""Application settings — mirrors autogen.net appsettings.json section-for-section.

Uses pydantic-settings to read from environment variables / .env file.
All storage substrate URIs, auth tokens, and tier configuration live here.

Section mapping (appsettings.json → Python):
    AppIdentity          → AppIdentitySettings
    Elasticsearch        → ElasticsearchSettings
    EmbeddingOptions     → EmbeddingSettings
    RerankingOptions     → RerankingSettings
    Cache                → CacheSettings
    LightRag             → LightRagSettings
    QnA                  → QnASettings (+ QnATierConfig)
    LlmQueryAuth         → AuthSettings
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Nested sub-settings — each mirrors a section in autogen.net's appsettings.json
# ---------------------------------------------------------------------------


class AppIdentitySettings(BaseModel):
    """Primary tenancy dimension — mirrors appsettings.json AppIdentity section.

    Every store, index, graph, and factory is keyed by app_id.
    """

    default_app_id: str = "neetpg"
    allowed_app_ids: list[str] = Field(
        default_factory=lambda: ["neetpg", "neetug", "mds", "ems"]
    )


class ElasticsearchSettings(BaseModel):
    """Elasticsearch connection — mirrors appsettings.json Elasticsearch section.

    This is the actual vector + text storage substrate (NOT Qdrant).
    """

    url: str = "http://localhost:9200"
    username: str | None = None
    password: str | None = None
    embedding_dim: int = 1024


class EmbeddingSettings(BaseModel):
    """Embedding provider — mirrors appsettings.json EmbeddingOptions section."""

    provider: str = "jina"
    base_url: str = "https://api.jina.ai/v1"
    default_model: str = "jina-embeddings-v3"
    api_key: str | None = None  # resolved from JINA_API_KEY env

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: str | None) -> str | None:
        """Resolve api_key from JINA_API_KEY env or .env if not set via EMBEDDING_OPTIONS__API_KEY."""
        # If a value was explicitly provided (even empty string), use it
        if v is not None:
            return v
        # Check os.environ first (set via export or system env)
        key = os.environ.get("JINA_API_KEY")
        if key:
            return key
        # Fall back to reading .env directly (pydantic-settings doesn't load arbitrary vars)
        env_path = os.environ.get("DOTENV_FILE", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("JINA_API_KEY="):
                        return line.split("=", 1)[1].strip()
        return None

    def model_post_init(self, __context: object) -> None:
        """Post-init hook — runs after __init__ even when defaults are used.
        pydantic v2 does NOT call field_validators with mode="before" when a field
        uses its default value. This post-init hook ensures JINA_API_KEY is resolved
        from .env even when EMBEDDING_OPTIONS__API_KEY is not set.
        """
        if self.api_key is None:
            # Check os.environ first
            key = os.environ.get("JINA_API_KEY")
            if key:
                object.__setattr__(self, "api_key", key)
                return
            # Fall back to reading .env file directly
            env_path = os.environ.get("DOTENV_FILE", ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("JINA_API_KEY="):
                            object.__setattr__(self, "api_key", line.split("=", 1)[1].strip())
                            return


class RerankingSettings(BaseModel):
    """Reranking provider — mirrors appsettings.json RerankingOptions section."""

    provider: str = "qwen"
    base_url: str = "http://home.bhakars.com:8077"  # llama.cpp default in source
    default_model: str = "Qwen/Qwen3-Reranker-4B"
    # Self-hosted Qwen typically needs no key; hosted Jina-reranker fallback does.
    api_key: str | None = None
    timeout_seconds: float = 30.0


class CacheSettings(BaseModel):
    """Embedded file-backed cache — mirrors appsettings.json Cache section.

    Mirrors Agentic.OpenLM/Clients/CachingClient.cs (LiteDB).
    """

    base_path: str = "./cache/OpenLM"
    rotate_daily: bool = True  # daily file: CachingClient_{yyyyMMdd}.db
    memory_size: int = 1000  # in-process TTLCache entries
    memory_ttl_seconds: int = 3600


class LightRagSettings(BaseModel):
    """LightRAG configuration — mirrors appsettings.json LightRag section.

    Chunk sizes match LightRagConfig.cs exactly.
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"
    chunk_token_size: int = 960  # matches LightRagConfig.cs:15
    chunk_overlap_token_size: int = 128  # matches LightRagConfig.cs:16
    tiktoken_model_name: str = "cl100k_base"  # matches LightRagConfig.cs:17
    entity_summary_to_max_tokens: int = 500
    entity_extract_max_gleaning: int = 1
    checkpoint_path: str = "./.checkpoints"


class QnATierConfig(BaseModel):
    """13 model roles per tier — mirrors QnAModels record in Program.cs."""

    explanation_model: str
    conversation_model: str
    thinking_model: str
    parallel_thinking_models: list[str] = Field(default_factory=list)
    method_call_model: str
    relevance_check_model: str
    segment_finder_model: str
    question_category_model: str
    action_dispatcher_model: str
    answer_extraction_model: str
    option_explanation_model: str
    detailed_explanation_model: str
    short_explanation_model: str
    hint_model: str


class QnASettings(BaseModel):
    """QnA configuration — mirrors appsettings.json QnA section."""

    tier_configurations: dict[str, QnATierConfig] = Field(default_factory=dict)
    models_catalog_path: str = "Configuration/models.json"


class AuthSettings(BaseModel):
    """Authentication — mirrors appsettings.json LlmQueryAuth section.

    Uses the exact header name from the .NET source: X-LlmQuery-Token.
    """

    header_name: str = "X-LlmQuery-Token"
    allowed_token: str | None = None  # resolved from LLMQUERY_TOKEN env
    smes: list[str] = Field(default_factory=list)
    testing_users: list[str] = Field(default_factory=list)


class RedisSettings(BaseModel):
    """Redis connection — used by arq job queue (background worker) and rate limiting."""

    url: str = "redis://localhost:6379"


class ConversationSettings(BaseModel):
    """Conversation persistence — SQLite for dev, asyncpg for prod."""

    database_url: str = "sqlite+aiosqlite:///./conversations.db"


class RateLimitSettings(BaseModel):
    """Per-route per-key rate limits (requests/minute).

    Calibrated to endpoint cost: cheap reads get more headroom, expensive
    or mutating operations get tighter caps.
    """

    qna_per_minute: int = 30
    ingest_per_minute: int = 5
    graph_query_per_minute: int = 30
    mcp_query_per_minute: int = 30
    mcp_question_per_minute: int = 10
    mcp_user_per_minute: int = 60


# ---------------------------------------------------------------------------
# Top-level Settings — reads from .env with __ nested delimiter
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Typed configuration — single source of truth for all env-derived values.

    Reads from .env file and/or environment variables at construction time.
    Every nested sub-model maps to a corresponding section in
    autogen.net's appsettings.json.

    Environment variable naming uses __ as the nested delimiter:
        APP_IDENTITY__DEFAULT_APP_ID=neetpg
        ELASTICSEARCH__URL=http://localhost:9200
        LLM_QUERY_AUTH__HEADER_NAME=X-LlmQuery-Token
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Environment ---
    env: str = "dev"

    # --- Nested sections (mirroring appsettings.json) ---
    app_identity: AppIdentitySettings = AppIdentitySettings()
    elasticsearch: ElasticsearchSettings = ElasticsearchSettings()
    embedding_options: EmbeddingSettings = EmbeddingSettings()
    reranking_options: RerankingSettings = RerankingSettings()
    cache: CacheSettings = CacheSettings()
    lightrag: LightRagSettings = LightRagSettings()
    qna: QnASettings = QnASettings()
    llm_query_auth: AuthSettings = AuthSettings()
    redis: RedisSettings = RedisSettings()
    rate_limits: RateLimitSettings = RateLimitSettings()
    conversation: ConversationSettings = ConversationSettings()
