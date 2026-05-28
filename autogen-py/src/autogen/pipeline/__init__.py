"""7-stage resumable ingestion pipeline (Phase 3 Day 15).

Produces the three Elasticsearch indices + Neo4j graph for an app_id from
raw documents.  Mirrors EntityExtractionPipeline.cs:67-107.

Public API:
    EntityExtractionPipeline  — orchestrator: run(app_id, docs) → populates all stores.
    CheckpointManager         — JSON-backed checkpoint file (pipeline.checkpoint.json).
    EntityRelationProcessor   — LLM summarizer for bloated entity/relation descriptions.
    VectorIndexer             — batches entity/relation/chunk embeddings to Elasticsearch.
"""

from __future__ import annotations

from autogen.pipeline.checkpoint import CheckpointManager
from autogen.pipeline.pipeline import EntityExtractionPipeline
from autogen.pipeline.processor import EntityRelationProcessor
from autogen.pipeline.vector_indexer import VectorIndexer

__all__ = [
    "CheckpointManager",
    "EntityExtractionPipeline",
    "EntityRelationProcessor",
    "VectorIndexer",
]