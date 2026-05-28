"""VectorIndexer — batch-embed and index entities, relations, and chunks.

Phase 3 Day 15 Stages 5-7.  Knows how to format each type for embedding:
    - Entity:     f"{entity_name}: {description}"
    - Relation:   f"{src_name} -> {tgt_name}: {description} (keywords: {kw})"
    - Chunk:      chunk.content (truncated to ~2048 chars)

All upserts go through the already-built ElasticVectorStore (Day 4),
which handles index creation, batch embedding, and bulk indexing.

Usage::

    indexer = VectorIndexer(embedding_client, vector_store_factory)
    await indexer.index_entities(entities, app_id="neetpg")
    await indexer.index_relations(relations, app_id="neetpg")
    await indexer.index_chunks(chunks, app_id="neetpg")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogen.embeddings.jina import JinaEmbeddingClient
    from autogen.models.storage import EntityNode, EntityRelation, TextChunk
    from autogen.storage.elastic import VectorStoreFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Max chars for chunk content before truncation
# ---------------------------------------------------------------------------
_MAX_CHUNK_TEXT_LEN = 2048


class VectorIndexer:
    """Batch-embed and upsert entities, relations, and chunks.

    Three methods, each batching at 32 items per embedding call (matching
    Jina's optimal batch size).
    """

    def __init__(
        self,
        embedding_client: JinaEmbeddingClient,
        store_factory: VectorStoreFactory,
    ) -> None:
        """
        Args:
            embedding_client: Jina embedding client for text batch embedding.
            store_factory: VectorStoreFactory to create per-type stores.
        """
        self._embed = embedding_client
        self._factory = store_factory
        self._batch_size = 32

    # ------------------------------------------------------------------
    # Entity embedding (Stage 5)
    # ------------------------------------------------------------------
    async def index_entities(
        self,
        entities: list[EntityNode],
        app_id: str,
    ) -> int:
        """Embed and upsert entities into entitynode_{appId}_1024.

        After summarization (EntityRelationProcessor), each entity's
        description is clean.  Embed text: f"{name}: {description}".

        Args:
            entities: List of merged+summarized EntityNode objects.
            app_id: Tenant identifier for index naming.

        Returns:
            Number of entities indexed.
        """
        if not entities:
            return 0

        from autogen.models.storage import EntityNode

        store = self._factory.create(app_id, EntityNode)

        # Build texts for embedding
        texts: list[str] = []
        for e in entities:
            text = f"{e.entity_name}: {e.description}" if e.description else e.entity_name
            texts.append(text)

        total = 0
        for i in range(0, len(entities), self._batch_size):
            batch = entities[i : i + self._batch_size]
            batch_texts = texts[i : i + self._batch_size]

            try:
                vectors = await self._embed.batch_embed(
                    batch_texts, task="retrieval.passage"
                )
                # Attach embeddings to entities
                for ent, vec in zip(batch, vectors):
                    ent.embedding = vec  # type: ignore[assignment]

                await store.upsert(batch)
                total += len(batch)
                logger.debug(
                    "index_entities batch %d/%d (%d items)",
                    (i // self._batch_size) + 1,
                    (len(entities) + self._batch_size - 1) // self._batch_size,
                    len(batch),
                )
            except Exception:
                logger.warning(
                    "index_entities batch starting at %d failed", i, exc_info=True
                )
                # Continue with next batch; don't lose the whole run

        logger.info(
            "index_entities complete: %d/%d entities embedded", total, len(entities)
        )
        return total

    # ------------------------------------------------------------------
    # Relation embedding (Stage 6)
    # ------------------------------------------------------------------
    async def index_relations(
        self,
        relations: list[EntityRelation],
        app_id: str,
    ) -> int:
        """Embed and upsert relations into entityrelation_{appId}_1024.

        Embed text:
            f"{source_name} -> {target_name}: {description} (keywords: {kws})"

        Args:
            relations: List of merged EntityRelation objects.
            app_id: Tenant identifier.

        Returns:
            Number of relations indexed.
        """
        if not relations:
            return 0

        from autogen.models.storage import EntityRelation

        store = self._factory.create(app_id, EntityRelation)

        texts: list[str] = []
        for r in relations:
            kw_str = ", ".join(r.keywords) if r.keywords else ""
            desc = r.description or ""
            text = f"{r.source_name} -> {r.target_name}: {desc}"
            if kw_str:
                text += f" (keywords: {kw_str})"
            texts.append(text)

        total = 0
        for i in range(0, len(relations), self._batch_size):
            batch = relations[i : i + self._batch_size]
            batch_texts = texts[i : i + self._batch_size]

            try:
                vectors = await self._embed.batch_embed(
                    batch_texts, task="retrieval.passage"
                )
                for rel, vec in zip(batch, vectors):
                    rel.embedding = vec  # type: ignore[assignment]

                await store.upsert(batch)
                total += len(batch)
            except Exception:
                logger.warning(
                    "index_relations batch starting at %d failed", i, exc_info=True
                )

        logger.info(
            "index_relations complete: %d/%d relations embedded",
            total,
            len(relations),
        )
        return total

    # ------------------------------------------------------------------
    # Chunk embedding (Stage 7)
    # ------------------------------------------------------------------
    async def index_chunks(
        self,
        chunks: list[TextChunk],
        app_id: str,
    ) -> int:
        """Embed and upsert chunks into textchunk_{appId}_1024.

        Embed text: chunk.content (truncated to _MAX_CHUNK_TEXT_LEN chars).

        Args:
            chunks: List of TextChunk objects with populated content.
            app_id: Tenant identifier.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        from autogen.models.storage import TextChunk

        store = self._factory.create(app_id, TextChunk)

        texts: list[str] = []
        for c in chunks:
            t = c.content or ""
            if len(t) > _MAX_CHUNK_TEXT_LEN:
                t = t[:_MAX_CHUNK_TEXT_LEN]
            texts.append(t)

        total = 0
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            batch_texts = texts[i : i + self._batch_size]

            try:
                vectors = await self._embed.batch_embed(
                    batch_texts, task="retrieval.passage"
                )
                for ch, vec in zip(batch, vectors):
                    ch.embedding = vec  # type: ignore[assignment]

                await store.upsert(batch)
                total += len(batch)
            except Exception:
                logger.warning(
                    "index_chunks batch starting at %d failed", i, exc_info=True
                )

        logger.info(
            "index_chunks complete: %d/%d chunks embedded", total, len(chunks)
        )
        return total