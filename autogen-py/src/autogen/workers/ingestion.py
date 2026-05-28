"""arq background worker for document ingestion — Day 20.

Runs the 7-stage EntityExtractionPipeline for a submitted batch of documents.
Started as a separate process alongside the web server:

    arq autogen.workers.ingestion.WorkerSettings

The web server enqueues jobs via::

    await arq_pool.enqueue_job("run_pipeline", app_id, documents)

and the worker picks them up here. Both processes share the same Redis
instance but run independently — the web server thread is never blocked by
the potentially 5–15 minute ingestion run.

Checkpoint design: if the worker crashes mid-pipeline, Phase 3's
CheckpointManager resumes from the last completed stage on restart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arq.connections import RedisSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job function
# ---------------------------------------------------------------------------


async def run_pipeline(
    ctx: dict[str, Any],
    app_id: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the full 7-stage ingestion pipeline for ``app_id``.

    Args:
        ctx:       arq worker context (contains Redis connection etc.)
        app_id:    Tenant/exam identifier (e.g. "neetpg", "mds")
        documents: List of serialised document dicts, each carrying
                   {"id", "content", "metadata", "app_id"}.

    Returns:
        dict with ``stages_completed`` count on success.
    """
    from autogen.config.settings import Settings
    from autogen.embeddings.jina import JinaEmbeddingClient
    from autogen.lightrag import LightRag
    from autogen.llm.builder import build_llm_stack
    from autogen.logging.setup import configure_logging
    from autogen.models.storage import FullDoc

    configure_logging("INFO")
    settings = Settings()

    logger.info(
        "worker.run_pipeline.start",
        extra={"app_id": app_id, "doc_count": len(documents)},
    )

    # Convert raw dicts to FullDoc — drop unknown fields (metadata), fill defaults
    docs: list[FullDoc] = [
        FullDoc(
            id=d["id"],
            content=d["content"],
            app_id=d.get("app_id", app_id),
            source=d.get("source", ""),
        )
        for d in documents
    ]

    # Build LLM stack (UsageTracking → Caching → LiteLLM)
    cache_dir = Path(settings.cache.base_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    llm_client = build_llm_stack(
        cache_dir=cache_dir,
        memory_size=settings.cache.memory_size,
        memory_ttl=settings.cache.memory_ttl_seconds,
    )

    embedding_client = JinaEmbeddingClient(settings.embedding_options)

    # Pick the extraction model — cheapest configured tier, or open-source fallback
    extraction_model = _resolve_extraction_model(settings)

    # LightRag.build() wires: chunker → extractor → normalizer → processor
    # → vector_indexer → graph_factory → pipeline + HybridRetrieval
    rag = LightRag.build(
        app_id=app_id,
        settings=settings,
        llm=llm_client,
        embedding=embedding_client,
        extraction_model=extraction_model,
    )

    stats = await rag.index(docs)

    chunks_done = stats.get("chunks", 0)
    entities_done = stats.get("entities", 0)
    relations_done = stats.get("relations", 0)

    logger.info(
        "worker.run_pipeline.done",
        extra={
            "app_id": app_id,
            "chunks": chunks_done,
            "entities": entities_done,
            "relations": relations_done,
        },
    )

    # Close embedding client so the worker doesn't leak connections between jobs
    try:
        await embedding_client.aclose()
    except Exception:
        pass

    return {
        "app_id": app_id,
        "stages_completed": 7,
        "chunks": chunks_done,
        "entities": entities_done,
        "relations": relations_done,
    }


def _resolve_extraction_model(settings: Any) -> str:
    """Pick the cheapest configured model for entity extraction.

    Falls back to a well-known open-source default if no tiers are configured.
    """
    tier_cfgs = settings.qna.tier_configurations
    for tier_name in ("Free", "Testing", "Regular"):
        cfg = tier_cfgs.get(tier_name)
        if cfg:
            return getattr(cfg, "method_call_model", None) or getattr(cfg, "conversation_model", None) or "groq/llama-3.1-70b-versatile"
    return "groq/llama-3.1-70b-versatile"


# ---------------------------------------------------------------------------
# Worker settings — read by arq at startup
# ---------------------------------------------------------------------------


def _redis_settings() -> RedisSettings:
    """Read Redis URL from Settings (honours .env)."""
    try:
        from autogen.config.settings import Settings
        s = Settings()
        return RedisSettings.from_dsn(s.redis.url)
    except Exception:
        return RedisSettings()  # defaults to localhost:6379


class WorkerSettings:
    """arq worker configuration.

    Start with::

        arq autogen.workers.ingestion.WorkerSettings
    """

    functions = [run_pipeline]
    redis_settings = _redis_settings()
    max_jobs = 4            # concurrent pipeline runs
    job_timeout = 3600      # 1 hour max per job
    keep_result = 86400     # keep job result in Redis for 24 h (status polling)
    retry_jobs = False      # pipelines are resumable via checkpoints; arq retry not needed
