"""Tests for EntityExtractionPipeline — Phase 3 Day 15.

All external I/O (LLM, Elasticsearch, Neo4j, file-system) is mocked so
tests run without any infrastructure.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.chunking.chunker import TextChunker
from autogen.models.storage import EntityNode, EntityRelation, FullDoc, TextChunk
from autogen.pipeline.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Fixtures — lightweight mocks for every dependency
# ---------------------------------------------------------------------------


def _mock_extractor(nodes=None, relations=None):
    """Return an EntityExtractor mock that always yields the given nodes/relations."""
    from autogen.extraction.extractor import ExtractionResponse

    result = ExtractionResponse(
        nodes=nodes or [
            EntityNode(
                id="ent-(aspirin)",
                entity_name="Aspirin",
                entity_type="Drug",
                description="An NSAID.",
                source_ids=["chunk-001"],
                app_id="test",
            )
        ],
        relations=relations or [],
        content_keywords=["aspirin"],
    )
    mock = MagicMock()
    mock.extract_from_chunk = AsyncMock(return_value=result)
    return mock


def _mock_normalizer():
    mock = MagicMock()
    async def _norm(nodes, **kwargs):
        return {}
    mock.normalize = _norm
    return mock


def _mock_processor():
    mock = MagicMock()
    mock.summarize_entity = AsyncMock()
    mock.summarize_relation = AsyncMock()
    return mock


def _mock_indexer():
    mock = MagicMock()
    mock.index_entities = AsyncMock()
    mock.index_relations = AsyncMock()
    mock.index_chunks = AsyncMock()
    return mock


def _mock_graph_factory():
    graph_store = MagicMock()
    graph_store.ensure_constraints = AsyncMock()
    graph_store.upsert_nodes = AsyncMock()
    graph_store.upsert_edges = AsyncMock()
    factory = MagicMock()
    factory.create = MagicMock(return_value=graph_store)
    return factory, graph_store


def _mock_kv_store():
    store = MagicMock()
    store.get = AsyncMock(return_value=None)
    store.upsert = AsyncMock()
    store.filter_keys = AsyncMock(return_value=[])
    store.get_all = AsyncMock(return_value={})
    store.keys = AsyncMock(return_value=[])
    store.namespace = "test_ns"
    return store


def _build_pipeline(tmp_path: Path):
    """Build a fully-mocked EntityExtractionPipeline in tmp_path."""
    from autogen.pipeline.pipeline import EntityExtractionPipeline

    chunker = TextChunker(chunk_token_size=50, chunk_overlap_token_size=5)
    extractor = _mock_extractor()
    normalizer = _mock_normalizer()
    checker = CheckpointManager(workspace_dir=tmp_path / "checkpoint", app_id="test")
    processor = _mock_processor()
    indexer = _mock_indexer()
    graph_factory, graph_store = _mock_graph_factory()
    docs_kv = _mock_kv_store()
    chunks_kv = _mock_kv_store()
    strings_kv = _mock_kv_store()

    pipeline = EntityExtractionPipeline(
        app_id="test",
        chunker=chunker,
        extractor=extractor,
        normalizer=normalizer,
        checker=checker,
        processor=processor,
        indexer=indexer,
        graph_factory=graph_factory,
        docs_kv=docs_kv,
        chunks_kv=chunks_kv,
        strings_kv=strings_kv,
        intermediate_dir=tmp_path / "intermediate",
    )
    return pipeline, graph_store, extractor


@pytest.fixture
def tmp_workspace(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# CheckpointManager tests (no external deps)
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_stage_not_complete_initially(self, tmp_path):
        ckpt = CheckpointManager(tmp_path, app_id="test")
        assert not ckpt.is_stage_complete("ProcessBookSegments")

    def test_mark_stage_complete(self, tmp_path):
        ckpt = CheckpointManager(tmp_path, app_id="test")
        ckpt.mark_stage_complete("ProcessBookSegments")
        assert ckpt.is_stage_complete("ProcessBookSegments")

    def test_complete_stages_persist_across_instances(self, tmp_path):
        ckpt1 = CheckpointManager(tmp_path, app_id="test")
        ckpt1.mark_stage_complete("MergeEntities")

        ckpt2 = CheckpointManager(tmp_path, app_id="test")
        assert ckpt2.is_stage_complete("MergeEntities")

    def test_reset_clears_stages(self, tmp_path):
        ckpt = CheckpointManager(tmp_path, app_id="test")
        ckpt.mark_stage_complete("ProcessBookSegments")
        ckpt.reset()
        assert not ckpt.is_stage_complete("ProcessBookSegments")

    def test_completed_stages_dict(self, tmp_path):
        ckpt = CheckpointManager(tmp_path, app_id="test")
        ckpt.mark_stage_complete("StageA")
        assert "StageA" in ckpt.completed_stages


# ---------------------------------------------------------------------------
# Pipeline run — smoke test (all stages complete)
# ---------------------------------------------------------------------------


class TestPipelineRun:
    @pytest.mark.asyncio
    async def test_run_returns_stats(self, tmp_workspace):
        pipeline, graph_store, extractor = _build_pipeline(tmp_workspace)
        docs = [FullDoc(id="doc-001", content="Aspirin inhibits COX enzymes.", app_id="test")]

        stats = await pipeline.run(docs)

        assert "chunks" in stats
        assert "entities" in stats
        assert "relations" in stats
        assert stats["chunks"] >= 1

    @pytest.mark.asyncio
    async def test_extractor_called_per_chunk(self, tmp_workspace):
        pipeline, graph_store, extractor = _build_pipeline(tmp_workspace)
        docs = [FullDoc(id="doc-001", content="Aspirin inhibits COX enzymes.", app_id="test")]

        await pipeline.run(docs)

        assert extractor.extract_from_chunk.call_count >= 1

    @pytest.mark.asyncio
    async def test_graph_store_constraints_ensured(self, tmp_workspace):
        pipeline, graph_store, _ = _build_pipeline(tmp_workspace)
        docs = [FullDoc(id="doc-001", content="Aspirin inhibits COX enzymes.", app_id="test")]

        await pipeline.run(docs)

        graph_store.ensure_constraints.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_doc_list_produces_zero_stats(self, tmp_workspace):
        pipeline, _, _ = _build_pipeline(tmp_workspace)
        stats = await pipeline.run([])
        assert stats["chunks"] == 0


# ---------------------------------------------------------------------------
# Checkpointing — completed stages are skipped on re-run
# ---------------------------------------------------------------------------


class TestCheckpointing:
    @pytest.mark.asyncio
    async def test_completed_stage_is_skipped(self, tmp_workspace):
        """If ProcessBookSegments is already checkpointed, chunking is skipped."""
        pipeline, _, extractor = _build_pipeline(tmp_workspace)
        # Pre-mark stage 1 as complete
        pipeline._checker.mark_stage_complete("ProcessBookSegments")

        docs = [FullDoc(id="doc-001", content="Aspirin inhibits COX.", app_id="test")]

        # Run — stage 1 should be skipped (no chunking calls to chunker)
        # Stage 2 will try to reload chunks from KV (returns [] via mock)
        await pipeline.run(docs)

        # Extractor should NOT be called (no chunks from KV mock)
        assert extractor.extract_from_chunk.call_count == 0
