"""Tests for the Elasticsearch vector store.

These tests use a mock Elasticsearch client to verify:
    - Index naming convention
    - ensure_index idempotency
    - upsert validation (requires embeddings)
    - search_by_vector and embedding_search
    - query_by_ids and delete
    - VectorStoreFactory creation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.models.storage import (
    BookSegment,
    EntityNode,
    EntityRelation,
    TextChunk,
)
from autogen.storage.elastic import (
    ElasticVectorStore,
    VectorStoreFactory,
    _build_index_name,
    _build_mappings,
)

# ---------------------------------------------------------------------------
# Unit: _build_index_name
# ---------------------------------------------------------------------------


class TestBuildIndexName:
    """Verify index naming mirrors ElasticVectorDb.cs:203."""

    def test_textchunk_neetpg(self) -> None:
        name = _build_index_name(TextChunk, "neetpg")
        assert name == "textchunk_neetpg_1024"

    def test_entitynode_mds(self) -> None:
        name = _build_index_name(EntityNode, "mds")
        assert name == "entitynode_mds_1024"

    def test_entityrelation_ems(self) -> None:
        name = _build_index_name(EntityRelation, "ems")
        assert name == "entityrelation_ems_1024"

    def test_booksegment_neetug(self) -> None:
        name = _build_index_name(BookSegment, "neetug")
        assert name == "booksegment_neetug_1024"

    def test_custom_dim(self) -> None:
        name = _build_index_name(TextChunk, "neetpg", dim=768)
        assert name == "textchunk_neetpg_768"

    def test_case_insensitive_app_id(self) -> None:
        name = _build_index_name(TextChunk, "NEETPG")
        assert name == "textchunk_neetpg_1024"


# ---------------------------------------------------------------------------
# Unit: _build_mappings
# ---------------------------------------------------------------------------


class TestBuildMappings:
    """Verify the Elasticsearch index mapping structure."""

    def test_default_dim(self) -> None:
        mappings = _build_mappings(1024)
        props = mappings["properties"]
        assert props["id"]["type"] == "keyword"
        assert props["app_id"]["type"] == "keyword"
        assert props["content"]["type"] == "text"
        assert props["embedding"]["type"] == "dense_vector"
        assert props["embedding"]["dims"] == 1024
        assert props["embedding"]["index"] is True
        assert props["embedding"]["similarity"] == "cosine"

    def test_custom_dim(self) -> None:
        mappings = _build_mappings(768)
        assert mappings["properties"]["embedding"]["dims"] == 768


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_es_client() -> MagicMock:
    """Create a mock Elasticsearch async client."""
    client = MagicMock()
    client.indices = MagicMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.bulk = AsyncMock(return_value={"errors": False, "items": []})
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "chunk-1",
                        "_score": 0.95,
                        "_source": {
                            "id": "chunk-1",
                            "content": "Test content 1",
                            "app_id": "neetpg",
                            "tokens": 10,
                            "source_id": "src-1",
                            "metadata": {},
                        },
                    },
                    {
                        "_id": "chunk-2",
                        "_score": 0.85,
                        "_source": {
                            "id": "chunk-2",
                            "content": "Test content 2",
                            "app_id": "neetpg",
                            "tokens": 20,
                            "source_id": "src-1",
                            "metadata": {},
                        },
                    },
                ]
            }
        }
    )
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {
                    "_id": "chunk-1",
                    "found": True,
                    "_source": {
                        "id": "chunk-1",
                        "content": "Test content 1",
                        "app_id": "neetpg",
                        "tokens_count": 10,
                        "full_doc_id": "src-1",
                        "order": 0,
                        "keywords": [],
                        "metadata": {},
                    },
                },
                {
                    "_id": "chunk-2",
                    "found": True,
                    "_source": {
                        "id": "chunk-2",
                        "content": "Test content 2",
                        "app_id": "neetpg",
                        "tokens_count": 20,
                        "full_doc_id": "src-1",
                        "order": 1,
                        "keywords": [],
                        "metadata": {},
                    },
                },
            ]
        }
    )
    return client


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """Create a mock embedding client."""
    client = MagicMock()
    client.embed = AsyncMock(return_value=[[0.1] * 1024])
    return client


@pytest.fixture
def store(mock_es_client: MagicMock, mock_embedding_client: MagicMock) -> ElasticVectorStore[TextChunk]:
    """Create an ElasticVectorStore for TextChunk with mocks."""
    return ElasticVectorStore[TextChunk](
        es_client=mock_es_client,
        embedding_client=mock_embedding_client,
        app_id="neetpg",
        model_type=TextChunk,
        dim=1024,
    )


# ---------------------------------------------------------------------------
# Tests: ElasticVectorStore
# ---------------------------------------------------------------------------


class TestElasticVectorStore:
    """Tests for the ElasticVectorStore class."""

    # --- Properties ---

    def test_index_name(self, store: ElasticVectorStore[TextChunk]) -> None:
        assert store.index_name == "textchunk_neetpg_1024"

    def test_app_id(self, store: ElasticVectorStore[TextChunk]) -> None:
        assert store.app_id == "neetpg"

    # --- ensure_index ---

    @pytest.mark.asyncio
    async def test_ensure_index_creates_when_not_exists(
        self, store: ElasticVectorStore[TextChunk], mock_es_client: MagicMock
    ) -> None:
        await store.ensure_index()
        mock_es_client.indices.exists.assert_awaited_once_with(index="textchunk_neetpg_1024")
        mock_es_client.indices.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_index_idempotent(
        self, store: ElasticVectorStore[TextChunk], mock_es_client: MagicMock
    ) -> None:
        await store.ensure_index()
        await store.ensure_index()  # second call
        mock_es_client.indices.exists.assert_awaited_once()  # only first call checks

    @pytest.mark.asyncio
    async def test_ensure_index_skips_when_exists(
        self, mock_es_client: MagicMock, mock_embedding_client: MagicMock
    ) -> None:
        mock_es_client.indices.exists = AsyncMock(return_value=True)
        s = ElasticVectorStore[TextChunk](
            es_client=mock_es_client,
            embedding_client=mock_embedding_client,
            app_id="neetpg",
            model_type=TextChunk,
        )
        await s.ensure_index()
        mock_es_client.indices.create.assert_not_awaited()

    # --- upsert ---

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self, store: ElasticVectorStore[TextChunk]) -> None:
        count = await store.upsert([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_upsert_requires_embedding(
        self, store: ElasticVectorStore[TextChunk]
    ) -> None:
        chunk = TextChunk(id="chunk-1", content="no embedding")
        with pytest.raises(ValueError, match="embedding"):
            await store.upsert([chunk])

    @pytest.mark.asyncio
    async def test_upsert_success(
        self, store: ElasticVectorStore[TextChunk], mock_es_client: MagicMock
    ) -> None:
        chunk = TextChunk(
            id="chunk-1",
            content="test",
            embedding=[0.1] * 1024,
        )
        count = await store.upsert([chunk])
        assert count == 1
        mock_es_client.bulk.assert_awaited_once()

    # --- search_by_vector ---

    @pytest.mark.asyncio
    async def test_search_by_vector_returns_results(
        self, store: ElasticVectorStore[TextChunk]
    ) -> None:
        vector = [0.1] * 1024
        results = await store.search_by_vector(vector, top_k=2)
        assert len(results) == 2
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], TextChunk)
        assert isinstance(results[0][1], float)
        assert results[0][0].id == "chunk-1"
        assert results[0][1] == 0.95

    @pytest.mark.asyncio
    async def test_search_by_vector_empty_results(
        self, mock_es_client: MagicMock, mock_embedding_client: MagicMock
    ) -> None:
        mock_es_client.search = AsyncMock(return_value={"hits": {"hits": []}})
        s = ElasticVectorStore[TextChunk](
            es_client=mock_es_client,
            embedding_client=mock_embedding_client,
            app_id="neetpg",
            model_type=TextChunk,
        )
        results = await s.search_by_vector([0.1] * 1024)
        assert results == []

    # --- embedding_search ---

    @pytest.mark.asyncio
    async def test_embedding_search_uses_embedding_client(
        self, store: ElasticVectorStore[TextChunk], mock_embedding_client: MagicMock
    ) -> None:
        results = await store.embedding_search("test query", top_k=2)
        assert len(results) == 2
        mock_embedding_client.embed.assert_awaited_once_with(["test query"], task="retrieval.query")

    @pytest.mark.asyncio
    async def test_embedding_search_raises_without_client(
        self, mock_es_client: MagicMock
    ) -> None:
        s = ElasticVectorStore[TextChunk](
            es_client=mock_es_client,
            embedding_client=None,
            app_id="neetpg",
            model_type=TextChunk,
        )
        with pytest.raises(RuntimeError, match="embedding client"):
            await s.embedding_search("test")

    # --- query_by_ids ---

    @pytest.mark.asyncio
    async def test_query_by_ids_returns_in_order(
        self, store: ElasticVectorStore[TextChunk]
    ) -> None:
        results = await store.query_by_ids(["chunk-1", "chunk-2"])
        assert len(results) == 2
        assert results[0].id == "chunk-1"
        assert results[1].id == "chunk-2"

    @pytest.mark.asyncio
    async def test_query_by_ids_empty(self, store: ElasticVectorStore[TextChunk]) -> None:
        results = await store.query_by_ids([])
        assert results == []

    # --- delete ---

    @pytest.mark.asyncio
    async def test_delete_success(
        self, store: ElasticVectorStore[TextChunk], mock_es_client: MagicMock
    ) -> None:
        count = await store.delete(["chunk-1", "chunk-2"])
        assert count == 2
        mock_es_client.bulk.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_empty(self, store: ElasticVectorStore[TextChunk]) -> None:
        count = await store.delete([])
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: VectorStoreFactory
# ---------------------------------------------------------------------------


class TestVectorStoreFactory:
    """Tests for the VectorStoreFactory."""

    def test_create_returns_correct_store(
        self, mock_es_client: MagicMock, mock_embedding_client: MagicMock
    ) -> None:
        factory = VectorStoreFactory(
            es_client=mock_es_client,
            embedding_client=mock_embedding_client,
            dim=1024,
        )
        store = factory.create("neetpg", TextChunk)
        assert isinstance(store, ElasticVectorStore)
        assert store.index_name == "textchunk_neetpg_1024"
        assert store.app_id == "neetpg"

    def test_create_multiple_types(
        self, mock_es_client: MagicMock, mock_embedding_client: MagicMock
    ) -> None:
        factory = VectorStoreFactory(
            es_client=mock_es_client,
            embedding_client=mock_embedding_client,
        )
        entities = factory.create("neetpg", EntityNode)
        relations = factory.create("neetpg", EntityRelation)
        chunks = factory.create("neetpg", TextChunk)
        assert entities.index_name == "entitynode_neetpg_1024"
        assert relations.index_name == "entityrelation_neetpg_1024"
        assert chunks.index_name == "textchunk_neetpg_1024"

    def test_create_multiple_tenants(
        self, mock_es_client: MagicMock, mock_embedding_client: MagicMock
    ) -> None:
        factory = VectorStoreFactory(
            es_client=mock_es_client,
            embedding_client=mock_embedding_client,
        )
        neetpg = factory.create("neetpg", TextChunk)
        mds = factory.create("mds", TextChunk)
        assert neetpg.index_name == "textchunk_neetpg_1024"
        assert mds.index_name == "textchunk_mds_1024"
        assert neetpg.index_name != mds.index_name  # tenant isolation

    def test_create_without_embedding_client(
        self, mock_es_client: MagicMock
    ) -> None:
        factory = VectorStoreFactory(es_client=mock_es_client)
        store = factory.create("neetpg", TextChunk)
        assert store._embedding is None
