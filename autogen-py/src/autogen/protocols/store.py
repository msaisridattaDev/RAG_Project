"""Vector store protocol — mirrors autogen.net IVectorStorage<T>.

Defines the interface that ``ElasticVectorStore[T]`` (and any future
swap-in like Qdrant/Pinecone) implements.

The Protocol is keyed by ``(app_id, namespace, dimension)`` and exposes the
six operations the plan calls out:

    ensure_index()           — create index if missing (idempotent)
    upsert(items)            — bulk-index pre-embedded items
    embedding_search(q, k)   — embed query, kNN search
    search_by_vector(v, k)   — kNN search with a pre-computed vector
    query_by_ids(ids)        — fetch by string IDs (preserves order)
    delete(ids)              — remove by string IDs

This matches ``ElasticVectorStore`` in ``autogen.storage.elastic`` exactly,
so the concrete class is structurally substitutable for ``VectorStore[T]``.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class VectorStore(Protocol[T]):
    """Generic vector store — mirrors ``IVectorStorage<T>`` in autogen.net.

    One instance per ``(app_id, model_type)`` pair. Tenant isolation is
    encoded in the index name (``{namespace}_{app_id}_{dimension}``).
    """

    @property
    def app_id(self) -> str:
        """The tenant/app identifier this store is bound to."""
        ...

    @property
    def index_name(self) -> str:
        """The Elasticsearch index name (``{namespace}_{app_id}_{dim}``)."""
        ...

    async def ensure_index(self) -> None:
        """Create the index if it does not exist (idempotent)."""
        ...

    async def upsert(self, items: list[T]) -> int:
        """Bulk-index pre-embedded items. Raises ``ValueError`` if any
        item is missing its ``embedding``.
        """
        ...

    async def embedding_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[T, float]]:
        """Embed the query on the fly, then return top-k ``(item, score)``."""
        ...

    async def search_by_vector(
        self,
        vector: list[float],
        top_k: int = 10,
    ) -> list[tuple[T, float]]:
        """kNN search with a pre-computed query vector."""
        ...

    async def query_by_ids(self, ids: list[str]) -> list[T]:
        """Fetch items by their string IDs, preserving input order."""
        ...

    async def delete(self, ids: list[str]) -> int:
        """Remove items by their string IDs. Returns the number deleted."""
        ...
