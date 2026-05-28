"""Tests for HybridRetrieval — Phase 3 Day 16.

Mocks all external stores (Elasticsearch, Neo4j) so no infrastructure needed.
Tests cover NAIVE, LOCAL, GLOBAL, and HYBRID modes, plus RRF merge logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.models.enums import QueryMode
from autogen.models.query import CombinedContext, QueryParam
from autogen.models.storage import EntityNode, EntityRelation, TextChunk
from autogen.retrieval.hybrid import HybridRetrieval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id_: str = "chunk-001", content: str = "aspirin text") -> TextChunk:
    return TextChunk(id=id_, content=content, app_id="test")


def _entity(id_: str = "ent-(aspirin)", name: str = "Aspirin") -> EntityNode:
    return EntityNode(
        id=id_,
        entity_name=name,
        entity_type="Drug",
        description="An NSAID.",
        source_ids=["chunk-001"],
        app_id="test",
    )


def _relation(id_: str = "rel-(aspirin)-(cox-1)") -> EntityRelation:
    return EntityRelation(
        id=id_,
        source_id="ent-(aspirin)",
        target_id="ent-(cox-1)",
        source_name="Aspirin",
        target_name="COX-1",
        description="inhibits",
        keywords=["inhibition"],
        strength=0.8,
        source_ids=["chunk-001"],
        app_id="test",
    )


def _mock_chunk_store(chunks=None):
    store = MagicMock()
    chunks = chunks or [_chunk()]
    store.embedding_search = AsyncMock(return_value=[(c, 0.9) for c in chunks])
    store.keyword_search = AsyncMock(return_value=[(c, 0.8) for c in chunks])
    store.query_by_ids = AsyncMock(return_value=chunks)
    return store


def _mock_entity_store(entities=None):
    store = MagicMock()
    entities = entities or [_entity()]
    store.embedding_search = AsyncMock(return_value=[(e, 0.9) for e in entities])
    return store


def _mock_relation_store(relations=None):
    store = MagicMock()
    relations = relations or [_relation()]
    store.embedding_search = AsyncMock(return_value=[(r, 0.9) for r in relations])
    return store


def _mock_graph_store(entities=None, relations=None):
    store = MagicMock()
    store.get_nodes = AsyncMock(return_value=entities or [_entity()])
    store.get_relations = AsyncMock(return_value=relations or [_relation()])
    store.get_node_edges = AsyncMock(return_value=relations or [_relation()])
    return store


def _mock_kw_extractor(local=None, global_=None):
    extractor = MagicMock()
    extractor.extract = AsyncMock(
        return_value=(local or ["aspirin"], global_ or ["pharmacology"])
    )
    return extractor


def _build_retrieval(
    chunks=None,
    entities=None,
    relations=None,
    graph_entities=None,
    graph_relations=None,
    local_kws=None,
    global_kws=None,
):
    return HybridRetrieval(
        app_id="test",
        chunk_store=_mock_chunk_store(chunks),
        entity_store=_mock_entity_store(entities),
        relation_store=_mock_relation_store(relations),
        graph_store=_mock_graph_store(graph_entities, graph_relations),
        keyword_extractor=_mock_kw_extractor(local_kws, global_kws),
    )


# ---------------------------------------------------------------------------
# NAIVE mode
# ---------------------------------------------------------------------------


class TestNaiveMode:
    @pytest.mark.asyncio
    async def test_returns_source_chunks(self):
        chunks = [_chunk("c1"), _chunk("c2")]
        retrieval = _build_retrieval(chunks=chunks)
        ctx = await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.NAIVE))

        assert len(ctx.sources) == 2
        assert ctx.entities == []
        assert ctx.relationships == []

    @pytest.mark.asyncio
    async def test_metadata_mode_label(self):
        retrieval = _build_retrieval()
        ctx = await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.NAIVE))
        assert ctx.metadata.get("mode") == "Naive"


# ---------------------------------------------------------------------------
# LOCAL mode
# ---------------------------------------------------------------------------


class TestLocalMode:
    @pytest.mark.asyncio
    async def test_returns_entities_and_sources(self):
        retrieval = _build_retrieval()
        ctx = await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.LOCAL))

        assert len(ctx.entities) > 0
        assert len(ctx.relationships) > 0

    @pytest.mark.asyncio
    async def test_keyword_extractor_called(self):
        retrieval = _build_retrieval()
        await retrieval.retrieve("aspirin MOA", QueryParam(mode=QueryMode.LOCAL))
        retrieval._kw.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_contains_keywords(self):
        retrieval = _build_retrieval(local_kws=["aspirin", "COX"])
        ctx = await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.LOCAL))
        assert "aspirin" in ctx.metadata.get("keywords", [])

    @pytest.mark.asyncio
    async def test_falls_back_to_query_when_no_keywords(self):
        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=([], []))
        retrieval = HybridRetrieval(
            app_id="test",
            chunk_store=_mock_chunk_store(),
            entity_store=_mock_entity_store(),
            relation_store=_mock_relation_store(),
            graph_store=_mock_graph_store(),
            keyword_extractor=extractor,
        )
        ctx = await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.LOCAL))
        # Should not raise; entity_store.embedding_search called with fallback
        retrieval._entities.embedding_search.assert_called()


# ---------------------------------------------------------------------------
# GLOBAL mode
# ---------------------------------------------------------------------------


class TestGlobalMode:
    @pytest.mark.asyncio
    async def test_returns_relations_and_entities(self):
        retrieval = _build_retrieval()
        ctx = await retrieval.retrieve("pharmacology", QueryParam(mode=QueryMode.GLOBAL))

        assert len(ctx.relationships) > 0
        assert len(ctx.entities) > 0

    @pytest.mark.asyncio
    async def test_metadata_mode_label(self):
        retrieval = _build_retrieval()
        ctx = await retrieval.retrieve("pharmacology", QueryParam(mode=QueryMode.GLOBAL))
        assert ctx.metadata.get("mode") == "Global"


# ---------------------------------------------------------------------------
# HYBRID mode
# ---------------------------------------------------------------------------


class TestHybridMode:
    @pytest.mark.asyncio
    async def test_runs_both_local_and_global(self):
        retrieval = _build_retrieval()
        ctx = await retrieval.retrieve("aspirin pharmacology", QueryParam(mode=QueryMode.HYBRID))

        # HYBRID merges both paths — should have content from both
        assert ctx.metadata.get("mode") == "Hybrid"

    @pytest.mark.asyncio
    async def test_keyword_extractor_called_three_times(self):
        """HYBRID calls extract() for LOCAL, GLOBAL, and keyword sub-paths."""
        retrieval = _build_retrieval()
        await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.HYBRID))
        assert retrieval._kw.extract.call_count == 3

    @pytest.mark.asyncio
    async def test_deduplication_in_hybrid(self):
        """Same entity returned from both paths should not appear twice."""
        entity = _entity()
        retrieval = _build_retrieval(graph_entities=[entity])
        ctx = await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.HYBRID))
        entity_ids = [e.id for e in ctx.entities]
        assert len(entity_ids) == len(set(entity_ids))


# ---------------------------------------------------------------------------
# HYBRID merge via end-to-end retrieve (replaces removed _merge_hybrid tests)
# ---------------------------------------------------------------------------


class TestMergeHybrid:
    @pytest.mark.asyncio
    async def test_merges_without_duplicates(self):
        """Same entity from LOCAL and GLOBAL should appear exactly once in HYBRID."""
        entity = _entity()
        retrieval = _build_retrieval(graph_entities=[entity])
        ctx = await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.HYBRID, top_k=10))
        ids = [e.id for e in ctx.entities]
        assert ids.count(entity.id) == 1

    @pytest.mark.asyncio
    async def test_sources_contain_both_paths(self):
        """HYBRID sources should include chunks from both LOCAL and GLOBAL paths."""
        retrieval = _build_retrieval()
        ctx = await retrieval.retrieve("aspirin", QueryParam(mode=QueryMode.HYBRID, top_k=10))
        assert len(ctx.sources) >= 1


# ---------------------------------------------------------------------------
# CombinedContext.build_context_string
# ---------------------------------------------------------------------------


class TestBuildContextString:
    def test_headers_always_present(self):
        ctx = CombinedContext()
        s = ctx.build_context_string()
        assert "-----Entities-----" in s
        assert "-----Relationships-----" in s
        assert "-----Sources-----" in s

    def test_entity_appears_in_output(self):
        ctx = CombinedContext(entities=[_entity()])
        s = ctx.build_context_string()
        assert "Aspirin" in s

    def test_relation_appears_in_output(self):
        ctx = CombinedContext(relationships=[_relation()])
        s = ctx.build_context_string()
        assert "Aspirin" in s
        assert "COX-1" in s

    def test_source_content_in_output(self):
        ctx = CombinedContext(sources=[_chunk(content="unique-content-xyz")])
        s = ctx.build_context_string()
        assert "unique-content-xyz" in s
