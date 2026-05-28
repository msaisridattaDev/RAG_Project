"""Elasticsearch Vector Store — mirrors autogen.net ElasticVectorDb.cs.

A generic vector store over the elasticsearch-py async client.
Parameterized by a Pydantic model type T for type-safe retrieval.

Index naming: {type_name.lower()}_{app_id.lower()}_{dim}
    - Mirrors ElasticVectorDb.cs:203: $"{typeof(T).Name.ToLower()}_{_appId.ToLower()}_{Dimension}"

Six methods:
    - ensure_index()        — Create index if not exists (idempotent)
    - upsert(items)         — Bulk-index vectors (items must have .embedding populated)
    - embedding_search(query, top_k) — Embed query on the fly, then kNN search
    - search_by_vector(vector, top_k) — Search with a pre-computed vector
    - query_by_ids(ids)     — Fetch specific items by their string IDs
    - delete(ids)           — Remove items by ID

Plus VectorStoreFactory for per-(type, app_id) store creation.
"""

from __future__ import annotations

from typing import TypeVar

from elasticsearch import AsyncElasticsearch  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel

from autogen.config.settings import Settings
from autogen.embeddings.jina import JinaEmbeddingClient
from autogen.logging.setup import get_logger

logger = get_logger("autogen.storage.elastic")

T_co = TypeVar("T_co", bound=BaseModel, covariant=True)

# ---------------------------------------------------------------------------
# Default dimension — matches jina-embeddings-v3 output
# ---------------------------------------------------------------------------

DEFAULT_DIM = 1024

# ---------------------------------------------------------------------------
# Elasticsearch index mapping template
# ---------------------------------------------------------------------------


def _build_mappings(dim: int) -> dict:
    """Build the Elasticsearch index mappings for a dense_vector index.

    Args:
        dim: The embedding dimension (default 1024 for jina-embeddings-v3).

    Returns:
        A mappings dict suitable for indices.create().
    """
    return {
        "properties": {
            "id": {"type": "keyword"},
            "app_id": {"type": "keyword"},
            "content": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": dim,
                "index": True,
                "similarity": "cosine",
            },
        }
    }


def _build_index_name(model_type: type[BaseModel], app_id: str, dim: int = DEFAULT_DIM) -> str:  # type: ignore[type-arg]
    """Build the Elasticsearch index name for a (type, app_id) pair.

    Mirrors ElasticVectorDb.cs:203:
        $"{typeof(T).Name.ToLower()}_{_appId.ToLower()}_{Dimension}"

    Args:
        model_type: The Pydantic model class (e.g., TextChunk, EntityNode).
        app_id: The tenant/app identifier (e.g., "neetpg", "mds").
        dim: The embedding dimension.

    Returns:
        An index name like "textchunk_neetpg_1024".
    """
    type_name = model_type.__name__.lower()
    return f"{type_name}_{app_id.lower()}_{dim}"


# ---------------------------------------------------------------------------
# ElasticVectorStore
# ---------------------------------------------------------------------------


class ElasticVectorStore[T_co]:
    """Generic vector store backed by Elasticsearch.

    One instance per (type, app_id) pair. Index name encodes both,
    providing static tenant isolation.

    Usage:
        store = ElasticVectorStore[TextChunk](
            es_client=es_client,
            embedding_client=embedding_client,
            app_id="neetpg",
            model_type=TextChunk,
            dim=1024,
        )
        await store.ensure_index()
        await store.upsert([chunk1, chunk2, ...])
        results = await store.embedding_search("what causes inflammation?", top_k=5)
    """

    def __init__(
        self,
        es_client: AsyncElasticsearch,
        embedding_client: JinaEmbeddingClient | None,
        app_id: str,
        model_type: type[T_co],
        dim: int = DEFAULT_DIM,
    ) -> None:
        """Initialize the vector store.

        Args:
            es_client: An elasticsearch-py async client instance.
            embedding_client: Embedding client for on-the-fly query embedding.
                Can be None if only search_by_vector is used.
            app_id: The tenant/app identifier.
            model_type: The Pydantic model class for type-safe deserialization.
            dim: The embedding dimension (default 1024).
        """
        self._client: AsyncElasticsearch = es_client
        self._embedding = embedding_client
        self._app_id = app_id
        self._model_type: type[T_co] = model_type
        self._dim = dim
        self._index_name = _build_index_name(model_type, app_id, dim)  # type: ignore[arg-type]
        self._ensured = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index_name(self) -> str:
        """The Elasticsearch index name for this store."""
        return self._index_name

    @property
    def app_id(self) -> str:
        """The tenant/app identifier."""
        return self._app_id

    # ------------------------------------------------------------------
    # ensure_index — lazy, idempotent
    # ------------------------------------------------------------------

    async def ensure_index(self) -> None:
        """Create the Elasticsearch index if it doesn't exist.

        Idempotent — safe to call multiple times. Only the first call
        pays a network round-trip.
        """
        if self._ensured:
            return

        exists = await self._client.indices.exists(index=self._index_name)
        if not exists:
            mappings = _build_mappings(self._dim)
            await self._client.indices.create(
                index=self._index_name,
                mappings=mappings,
            )
            logger.info(
                "elastic.index.created",
                index=self._index_name,
                dim=self._dim,
                model_type=self._model_type.__name__,
                app_id=self._app_id,
            )
        else:
            logger.debug(
                "elastic.index.exists",
                index=self._index_name,
            )

        self._ensured = True

    # ------------------------------------------------------------------
    # upsert — bulk-index vectors
    # ------------------------------------------------------------------

    async def upsert(self, items: list[T_co]) -> int:
        """Bulk-index vectors into Elasticsearch.

        Each item MUST have an .embedding field populated (list[float]).
        Raises ValueError if any item is missing its embedding.

        Args:
            items: The items to index. Each must have .id, .embedding, and
                be serializable to a dict via .model_dump().

        Returns:
            The number of items successfully upserted.

        Raises:
            ValueError: If any item is missing its embedding.
        """
        if not items:
            return 0

        # Validate embeddings are present
        for item in items:
            embedding = getattr(item, "embedding", None)
            if not embedding:
                msg = (
                    "upsert requires items with .embedding populated; "
                    f"item {getattr(item, 'id', '<unknown>')} has no embedding. "
                    "Use VectorIndexer helper to embed before upsert."
                )
                raise ValueError(msg)

        await self.ensure_index()

        # Build bulk body
        # Elasticsearch bulk API format: alternating action + document lines
        operations: list[dict] = []
        for item in items:
            doc = item.model_dump()  # type: ignore[attr-defined]
            # Remove embedding from _source (stored in _source but we keep it)
            # Actually we keep embedding in _source for re-indexing purposes
            operations.append({"index": {"_index": self._index_name, "_id": item.id}})  # type: ignore[attr-defined]
            operations.append(doc)

        response = await self._client.bulk(operations=operations, refresh=True)

        if response.get("errors"):
            error_count = sum(
                1 for item in response.get("items", []) if "error" in item.get("index", {})
            )
            logger.error(
                "elastic.upsert.errors",
                index=self._index_name,
                total=len(items),
                errors=error_count,
            )
        else:
            logger.info(
                "elastic.upsert.ok",
                index=self._index_name,
                count=len(items),
            )

        return len(items)

    # ------------------------------------------------------------------
    # embedding_search — embed query, then kNN search
    # ------------------------------------------------------------------

    async def embedding_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[T_co, float]]:
        """Embed the query string on the fly, then run kNN search.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            A list of (item, score) tuples sorted by score descending.

        Raises:
            RuntimeError: If no embedding client is configured.
        """
        if self._embedding is None:
            msg = (
                "embedding_search requires an embedding client; "
                "configure one at construction time or use search_by_vector instead."
            )
            raise RuntimeError(msg)

        vectors = await self._embedding.embed([query], task="retrieval.query")
        if not vectors:
            return []

        return await self.search_by_vector(vectors[0], top_k=top_k)

    # ------------------------------------------------------------------
    # search_by_vector — search with a pre-computed vector
    # ------------------------------------------------------------------

    async def search_by_vector(
        self,
        vector: list[float],
        top_k: int = 10,
    ) -> list[tuple[T_co, float]]:
        """Search the index with a pre-computed embedding vector.

        Uses Elasticsearch kNN search with HNSW + cosine similarity.

        Args:
            vector: The query embedding vector (1024 floats).
            top_k: Number of results to return.

        Returns:
            A list of (item, score) tuples sorted by score descending.
        """
        await self.ensure_index()

        # Over-fetch internally for better recall (num_candidates = top_k * 20)
        num_candidates = max(top_k * 20, 100)

        response = await self._client.search(
            index=self._index_name,
            knn={
                "field": "embedding",
                "query_vector": vector,
                "k": top_k,
                "num_candidates": num_candidates,
            },
            size=top_k,
            source_excludes=["embedding"],  # strip the 1024-float field from results
        )

        hits = response.get("hits", {}).get("hits", [])
        results: list[tuple[T_co, float]] = []

        for hit in hits:
            source = hit["_source"]
            score = float(hit["_score"])
            # Reconstruct the Pydantic model from source
            item: T_co = self._model_type.model_validate(source)  # type: ignore[assignment]
            results.append((item, score))

        logger.debug(
            "elastic.search.ok",
            index=self._index_name,
            top_k=top_k,
            hits=len(results),
        )

        return results

    # ------------------------------------------------------------------
    # query_by_ids — fetch specific items by their string IDs
    # ------------------------------------------------------------------

    async def query_by_ids(self, ids: list[str]) -> list[T_co]:
        """Fetch specific items by their original string IDs.

        Preserves input order.

        Args:
            ids: The string IDs to fetch.

        Returns:
            A list of deserialized model instances in input order.
            Missing IDs are silently omitted.
        """
        if not ids:
            return []

        await self.ensure_index()

        response = await self._client.mget(
            index=self._index_name,
            body={"ids": ids},
            source_excludes=["embedding"],
        )

        # Build a dict of id -> item for ordering
        items_by_id: dict[str, T_co] = {}
        for doc in response.get("docs", []):
            if doc.get("found"):
                source = doc["_source"]
                item: T_co = self._model_type.model_validate(source)  # type: ignore[assignment]
                items_by_id[doc["_id"]] = item

        # Return in input order
        return [items_by_id[id_] for id_ in ids if id_ in items_by_id]

    # ------------------------------------------------------------------
    # delete — remove items by ID
    # ------------------------------------------------------------------

    async def delete(self, ids: list[str]) -> int:
        """Remove items from the index by their string IDs.

        Args:
            ids: The string IDs to remove.

        Returns:
            The number of items successfully deleted.
        """
        if not ids:
            return 0

        await self.ensure_index()

        response = await self._client.bulk(
            operations=[
                {"delete": {"_index": self._index_name, "_id": id_}}
                for id_ in ids
            ],
            refresh=True,
        )

        deleted = len(ids)
        if response.get("errors"):
            error_count = sum(
                1 for item in response.get("items", []) if "error" in item.get("delete", {})
            )
            logger.error(
                "elastic.delete.errors",
                index=self._index_name,
                requested=len(ids),
                errors=error_count,
            )
            deleted = len(ids) - error_count

        logger.info(
            "elastic.delete.ok",
            index=self._index_name,
            deleted=deleted,
        )

        return deleted


# ---------------------------------------------------------------------------
# VectorStoreFactory — mirrors IVectorStorageFactory
# ---------------------------------------------------------------------------


class VectorStoreFactory:
    """Factory for creating per-(type, app_id) ElasticVectorStore instances.

    Mirrors autogen.net's IVectorStorageFactory (RegisterServices.cs:119-121).

    Usage:
        factory = VectorStoreFactory(es_client, embedding_client, dim=1024)
        entities_store = factory.create("neetpg", EntityNode)
        chunks_store = factory.create("neetpg", TextChunk)
    """

    def __init__(
        self,
        es_client: AsyncElasticsearch,
        embedding_client: JinaEmbeddingClient | None = None,
        dim: int = DEFAULT_DIM,
    ) -> None:
        """Initialize the factory.

        Args:
            es_client: An elasticsearch-py async client instance.
            embedding_client: Optional embedding client for on-the-fly query embedding.
            dim: The embedding dimension (default 1024).
        """
        self._client: AsyncElasticsearch = es_client
        self._embedding = embedding_client
        self._dim = dim

    def create(self, app_id: str, model_type: type[T_co]) -> ElasticVectorStore[T_co]:
        """Create a vector store for the given (app_id, model_type) pair.

        Args:
            app_id: The tenant/app identifier.
            model_type: The Pydantic model class.

        Returns:
            An ElasticVectorStore bound to the correct index.
        """
        return ElasticVectorStore[T_co](
            es_client=self._client,
            embedding_client=self._embedding,
            app_id=app_id,
            model_type=model_type,
            dim=self._dim,
        )


# ---------------------------------------------------------------------------
# AsyncElasticsearch client factory
# ---------------------------------------------------------------------------


def _create_es_client(settings: Settings) -> AsyncElasticsearch:
    """Build an AsyncElasticsearch client from Settings.elasticsearch.

    Honors ELASTICSEARCH__URL, ELASTICSEARCH__USERNAME, ELASTICSEARCH__PASSWORD.
    Auth is sent only when a non-empty username is configured.
    """
    cfg = settings.elasticsearch
    basic_auth = (cfg.username, cfg.password) if (cfg.username and cfg.password) else None
    return AsyncElasticsearch(
        hosts=[cfg.url],
        basic_auth=basic_auth,
        verify_certs=False,
        request_timeout=60,
    )
