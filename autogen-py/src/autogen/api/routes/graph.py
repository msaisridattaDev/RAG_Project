"""Graph ingest + status endpoints — Day 20.

POST /v1/graph/{app_id}/ingest
    Enqueue the 7-stage EntityExtractionPipeline as an arq background job.
    Returns 202 immediately with a job_id; actual work runs in the worker process.

GET /v1/graph/{app_id}/ingest/{job_id}
    Poll the status of a previously submitted ingest job.

The graph query endpoint (/v1/{app_id}/query) already exists in query.py;
this router handles only the async ingest surface.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from autogen.api.deps import get_settings
from autogen.api.limiter import limiter
from autogen.config.settings import Settings
from autogen.logging.setup import get_logger

router = APIRouter(prefix="/v1/graph", tags=["graph"])
logger = get_logger("autogen.api.graph")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class DocumentInput(BaseModel):
    """Single document submitted for ingestion."""

    id: str = Field(..., description="Stable content-addressed or caller-provided ID")
    content: str = Field(..., min_length=1, description="Full text content to ingest")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Body for POST /v1/graph/{app_id}/ingest."""

    documents: list[DocumentInput] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    job_id: str
    status: str = "queued"
    document_count: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | in_progress | complete | failed | not_found
    stages_completed: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# POST /v1/graph/{app_id}/ingest
# ---------------------------------------------------------------------------


@router.post("/{app_id}/ingest", response_model=IngestResponse, status_code=202)
@limiter.limit("5/minute")
async def ingest_documents(
    app_id: str,
    body: IngestRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Submit documents for async ingestion into the {app_id} knowledge graph.

    The 7-stage pipeline (chunking → entity extraction → merge → embed →
    store into Elasticsearch + Neo4j) runs in the arq worker process.
    This endpoint returns immediately with a job_id.

    Example::

        curl -X POST localhost:8000/v1/graph/neetpg/ingest \\
            -H "X-LlmQuery-Token: $TOKEN" \\
            -H "Content-Type: application/json" \\
            -d '{"documents":[{"id":"d1","content":"Aspirin inhibits COX-1..."}]}'
    """
    if app_id not in settings.app_identity.allowed_app_ids:
        raise HTTPException(status_code=404, detail=f"unknown app_id: {app_id!r}")

    arq_pool = getattr(request.app.state, "arq_pool", None)
    if arq_pool is None:
        raise HTTPException(
            status_code=503,
            detail="arq job queue unavailable — check Redis connection",
        )

    # Stamp app_id on each document before serialising into the queue
    docs_payload = [
        {"id": doc.id, "content": doc.content, "metadata": doc.metadata, "app_id": app_id}
        for doc in body.documents
    ]

    job = await arq_pool.enqueue_job("run_pipeline", app_id, docs_payload)
    job_id: str = job.job_id

    logger.info(
        "graph.ingest.queued",
        app_id=app_id,
        job_id=job_id,
        doc_count=len(body.documents),
    )

    return IngestResponse(
        job_id=job_id,
        status="queued",
        document_count=len(body.documents),
    )


# ---------------------------------------------------------------------------
# GET /v1/graph/{app_id}/ingest/{job_id}
# ---------------------------------------------------------------------------


@router.get("/{app_id}/ingest/{job_id}", response_model=JobStatusResponse)
async def get_ingest_status(
    app_id: str,
    job_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> JobStatusResponse:
    """Poll the status of a previously submitted ingest job.

    Possible statuses:
        queued       — waiting in Redis queue
        in_progress  — worker picked it up, pipeline running
        complete     — all 7 stages finished successfully
        failed       — pipeline raised an unhandled exception
        not_found    — job_id unknown (expired or never existed)
    """
    if app_id not in settings.app_identity.allowed_app_ids:
        raise HTTPException(status_code=404, detail=f"unknown app_id: {app_id!r}")

    arq_pool = getattr(request.app.state, "arq_pool", None)
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="arq job queue unavailable")

    try:
        from arq.jobs import Job, JobStatus

        job = Job(job_id=job_id, redis=arq_pool)
        raw_status = await job.status()

        if raw_status == JobStatus.not_found:
            return JobStatusResponse(job_id=job_id, status="not_found")

        if raw_status == JobStatus.queued:
            return JobStatusResponse(job_id=job_id, status="queued")

        if raw_status == JobStatus.in_progress:
            return JobStatusResponse(job_id=job_id, status="in_progress")

        if raw_status == JobStatus.complete:
            result = await job.result(timeout=0.1)
            stages = result.get("stages_completed") if isinstance(result, dict) else None
            return JobStatusResponse(
                job_id=job_id, status="complete", stages_completed=stages
            )

        # deferred or other states — treat as queued
        return JobStatusResponse(job_id=job_id, status="queued")

    except Exception as exc:
        logger.error("graph.ingest.status_error", job_id=job_id, error=str(exc))
        return JobStatusResponse(job_id=job_id, status="failed", error=str(exc))
