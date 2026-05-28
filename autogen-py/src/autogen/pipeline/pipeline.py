"""EntityExtractionPipeline — 7-stage resumable ingestion pipeline.

Phase 3 Day 15. Mirrors EntityExtractionPipeline.cs:67-107 exactly.
Orchestrates the entire document → knowledge-graph flow for a single app_id.

Stages:
    1. ProcessBookSegments      — chunk documents, persist to KV
    2. ExtractEntitiesAndRelations — per-chunk LLM extraction with gleaning
    3. MergeEntities            — cross-chunk entity dedup + name normalization
    4. MergeRelations           — cross-chunk relation merge + strength accumulation
    5. GenerateEntityDescriptionsAndEmbeddings — EntityRelationProcessor summarization
                                                  + embed to entitynode_{appId}_1024
    6. GenerateRelationDescriptionsAndEmbeddings — same for edges → entityrelation_{appId}_1024
    7. StoreIntoDatabases       — Neo4j upsert (namespace=appId) + chunk embeddings
                                  → textchunk_{appId}_1024

Each stage writes a checkpoint after completion. On restart, completed stages
are skipped — you only pay for the work that wasn't finished.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogen.chunking.chunker import TextChunker
    from autogen.extraction.extractor import EntityExtractor
    from autogen.extraction.normalizer import NormalizeNames
    from autogen.models.storage import (
        EntityNode,
        EntityRelation,
        FullDoc,
        TextChunk,
    )
    from autogen.pipeline.checkpoint import CheckpointManager
    from autogen.pipeline.processor import EntityRelationProcessor
    from autogen.pipeline.vector_indexer import VectorIndexer
    from autogen.protocols.graph import GraphStore
    from autogen.protocols.kvstore import KeyValueStore
    from autogen.protocols.store import GraphStoreFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency cap for extraction (mirrors .NET semaphore of 4)
# ---------------------------------------------------------------------------
_EXTRACTION_CONCURRENCY = 4


class EntityExtractionPipeline:
    """7-stage resumable ingestion pipeline for one app_id.

    Usage::

        pipeline = EntityExtractionPipeline(
            app_id="neetpg",
            chunker=...,
            extractor=...,
            normalizer=...,
            checker=...,
            processor=...,
            indexer=...,
            graph_factory=...,
            docs_kv=...,
            chunks_kv=...,
            strings_kv=...,
            intermediate_dir=Path("./pipeline_state"),
        )
        await pipeline.run(documents)
    """

    def __init__(
        self,
        *,
        app_id: str,
        chunker: TextChunker,
        extractor: EntityExtractor,
        normalizer: NormalizeNames,
        checker: CheckpointManager,
        processor: EntityRelationProcessor,
        indexer: VectorIndexer,
        graph_factory: GraphStoreFactory,
        docs_kv: KeyValueStore[FullDoc],
        chunks_kv: KeyValueStore[TextChunk],
        strings_kv: KeyValueStore[str],
        intermediate_dir: Path,
        extraction_concurrency: int = _EXTRACTION_CONCURRENCY,
    ) -> None:
        self._app_id = app_id
        self._chunker = chunker
        self._extractor = extractor
        self._normalizer = normalizer
        self._checker = checker
        self._processor = processor
        self._indexer = indexer
        self._graph_factory = graph_factory
        self._docs_kv = docs_kv
        self._chunks_kv = chunks_kv
        self._strings_kv = strings_kv
        self._dir = intermediate_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(extraction_concurrency or 1)

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------
    async def run(self, documents: list[FullDoc]) -> dict[str, int]:
        """Run all 7 stages.  Returns summary counts.

        Args:
            documents: List of FullDoc records to ingest for this app_id.

        Returns:
            Dict with keys: chunks, entities, relations.
        """
        stats: dict[str, int] = {}

        # Stage 1
        chunks = await self._stage1_process_book_segments(documents)
        stats["chunks"] = len(chunks)

        # Stage 2
        extracted = await self._stage2_extract_entities_and_relations(chunks)
        stats["entities"] = sum(len(r["nodes"]) for r in extracted)
        stats["relations"] = sum(len(r["edges"]) for r in extracted)

        # Stage 3
        merged_entities = await self._stage3_merge_entities(extracted)

        # Stage 4
        merged_relations = await self._stage4_merge_relations(
            extracted, merged_entities
        )

        # Stage 5
        await self._stage5_generate_entity_descriptions_and_embeddings(
            merged_entities
        )

        # Stage 6
        await self._stage6_generate_relation_descriptions_and_embeddings(
            merged_relations
        )

        # Stage 7
        await self._stage7_store_into_databases(
            merged_entities, merged_relations, chunks
        )

        return stats

    # ------------------------------------------------------------------
    # Stage 1 — ProcessBookSegments
    # ------------------------------------------------------------------
    async def _stage1_process_book_segments(
        self, documents: list[FullDoc]
    ) -> list[TextChunk]:
        stage = "ProcessBookSegments"
        if self._checker.is_stage_complete(stage):
            logger.info("[%s] Skipping — already complete.", stage)
            # Re-load existing chunks for downstream stages
            return await self._load_existing_chunks()

        logger.info("[%s] Chunking %d documents …", stage, len(documents))
        all_chunks: list[TextChunk] = []

        for doc in documents:
            # Skip already-ingested docs
            existing = await self._docs_kv.get(doc.id)
            if existing is not None:
                logger.debug(
                    "Doc %s already in KV; skipping chunking.", doc.id
                )
                chunks_for_doc = await self._load_chunks_for_doc(doc.id)
                all_chunks.extend(chunks_for_doc)
                continue

            # Chunk the document — chunk_text() returns fully-initialized TextChunks
            chunks_for_doc = self._chunker.chunk_text(
                doc.content, app_id=self._app_id, full_doc_id=doc.id
            )

            # Persist chunks
            for c in chunks_for_doc:
                if await self._chunks_kv.get(c.id) is None:
                    await self._chunks_kv.upsert(c.id, c)
                if await self._strings_kv.get(c.id) is None:
                    await self._strings_kv.upsert(c.id, c.content)

            # Persist the doc record
            await self._docs_kv.upsert(doc.id, doc)
            all_chunks.extend(chunks_for_doc)

        self._checker.mark_stage_complete(stage)
        logger.info("[%s] Complete — %d chunks across %d docs.", stage, len(all_chunks), len(documents))
        return all_chunks

    # ------------------------------------------------------------------
    # Stage 2 — ExtractEntitiesAndRelations
    # ------------------------------------------------------------------
    async def _stage2_extract_entities_and_relations(
        self, chunks: list[TextChunk]
    ) -> list[dict]:
        stage = "ExtractEntitiesAndRelations"
        if self._checker.is_stage_complete(stage):
            logger.info("[%s] Skipping — already complete.", stage)
            return self._load_intermediate_json("extracted.json")

        logger.info("[%s] Extracting from %d chunks (concurrency=%d) …", stage, len(chunks), _EXTRACTION_CONCURRENCY)

        async def _extract_one(c: TextChunk) -> dict:
            async with self._sem:
                result = await self._extractor.extract_from_chunk(c)
            return {
                "chunk_id": c.id,
                "nodes": [n.model_dump() for n in result.nodes],
                "edges": [e.model_dump() for e in result.relations],
                "keywords": result.content_keywords,
            }

        tasks = [_extract_one(c) for c in chunks]
        results = await asyncio.gather(*tasks)
        extracted = [r for r in results if r is not None]

        self._save_intermediate_json(extracted, "extracted.json")
        self._checker.mark_stage_complete(stage)
        logger.info("[%s] Complete — %d extractions.", stage, len(extracted))
        return extracted

    # ------------------------------------------------------------------
    # Stage 3 — MergeEntities
    # ------------------------------------------------------------------
    async def _stage3_merge_entities(
        self, extracted: list[dict]
    ) -> list[EntityNode]:
        from autogen.models.storage import EntityNode

        stage = "MergeEntities"
        if self._checker.is_stage_complete(stage):
            logger.info("[%s] Skipping — already complete.", stage)
            data = self._load_intermediate_json("merged_entities.json")
            return [EntityNode(**d) for d in data]

        logger.info("[%s] Merging entities …", stage)

        # --- Collapse by entity ID ---
        merged: dict[str, EntityNode] = {}
        for ext in extracted:
            for raw in ext["nodes"]:
                node = EntityNode(**raw)
                if node.id in merged:
                    existing = merged[node.id]
                    # Accumulate descriptions (dedup)
                    for desc in node.descriptions:
                        if desc not in existing.descriptions:
                            existing.descriptions.append(desc)
                    # Union source IDs
                    for sid in node.source_ids:
                        if sid not in existing.source_ids:
                            existing.source_ids.append(sid)
                    # Union historical types
                    for ht in node.historical_entity_types:
                        if ht not in existing.historical_entity_types:
                            existing.historical_entity_types.append(ht)
                    # Rebuild combined description
                    existing.description = "\n".join(existing.descriptions)
                else:
                    merged[node.id] = node

        entities = list(merged.values())

        # --- Run global name normalization ---
        all_names = [e.entity_name for e in entities]
        if all_names:
            name_map = await self._normalizer.normalize(all_names)
            self._apply_name_map_to_entities(entities, name_map)

        self._save_intermediate_json(
            [e.model_dump() for e in entities], "merged_entities.json"
        )
        self._checker.mark_stage_complete(stage)
        logger.info("[%s] Complete — %d unique entities.", stage, len(entities))
        return entities

    # ------------------------------------------------------------------
    # Stage 4 — MergeRelations
    # ------------------------------------------------------------------
    async def _stage4_merge_relations(
        self,
        extracted: list[dict],
        merged_entities: list[EntityNode],
    ) -> list[EntityRelation]:
        from autogen.models.storage import EntityRelation

        stage = "MergeRelations"
        if self._checker.is_stage_complete(stage):
            logger.info("[%s] Skipping — already complete.", stage)
            data = self._load_intermediate_json("merged_relations.json")
            return [EntityRelation(**d) for d in data]

        logger.info("[%s] Merging relations …", stage)

        # Build entity name → entity lookups
        entity_by_name: dict[str, EntityNode] = {}
        for e in merged_entities:
            entity_by_name[e.entity_name.lower()] = e

        merged: dict[str, EntityRelation] = {}
        for ext in extracted:
            for raw in ext["edges"]:
                rel = EntityRelation(**raw)

                # Re-key via canonical entity names if available
                src_canonical = entity_by_name.get(rel.source_name.lower())
                tgt_canonical = entity_by_name.get(rel.target_name.lower())
                if src_canonical and tgt_canonical:
                    rel.source_id = src_canonical.id
                    rel.target_id = tgt_canonical.id
                    rel.source_name = src_canonical.entity_name
                    rel.target_name = tgt_canonical.entity_name
                    rel.id = EntityRelation.id_from_names(
                        src_canonical.entity_name, tgt_canonical.entity_name
                    )

                if rel.id in merged:
                    existing = merged[rel.id]
                    for desc in rel.descriptions:
                        if desc not in existing.descriptions:
                            existing.descriptions.append(desc)
                    for sid in rel.source_ids:
                        if sid not in existing.source_ids:
                            existing.source_ids.append(sid)
                    existing.keywords = list(
                        set(existing.keywords + rel.keywords)
                    )
                    existing.strength += rel.strength
                    existing.description = "\n".join(existing.descriptions)
                else:
                    merged[rel.id] = rel

        relations = list(merged.values())

        self._save_intermediate_json(
            [r.model_dump() for r in relations], "merged_relations.json"
        )
        self._checker.mark_stage_complete(stage)
        logger.info("[%s] Complete — %d unique relations.", stage, len(relations))
        return relations

    # ------------------------------------------------------------------
    # Stage 5 — GenerateEntityDescriptionsAndEmbeddings
    # ------------------------------------------------------------------
    async def _stage5_generate_entity_descriptions_and_embeddings(
        self, entities: list[EntityNode]
    ) -> None:
        stage = "GenerateEntityDescriptionsAndEmbeddings"
        if self._checker.is_stage_complete(stage):
            logger.info("[%s] Skipping — already complete.", stage)
            return

        logger.info("[%s] Summarizing + embedding %d entities …", stage, len(entities))

        # Summarize bloated descriptions
        for entity in entities:
            await self._processor.summarize_entity(entity)

        # Embed to Elasticsearch
        await self._indexer.index_entities(entities, self._app_id)

        self._checker.mark_stage_complete(stage)
        logger.info("[%s] Complete.", stage)

    # ------------------------------------------------------------------
    # Stage 6 — GenerateRelationDescriptionsAndEmbeddings
    # ------------------------------------------------------------------
    async def _stage6_generate_relation_descriptions_and_embeddings(
        self, relations: list[EntityRelation]
    ) -> None:
        stage = "GenerateRelationDescriptionsAndEmbeddings"
        if self._checker.is_stage_complete(stage):
            logger.info("[%s] Skipping — already complete.", stage)
            return

        logger.info("[%s] Summarizing + embedding %d relations …", stage, len(relations))

        for rel in relations:
            await self._processor.summarize_relation(rel)

        await self._indexer.index_relations(relations, self._app_id)

        self._checker.mark_stage_complete(stage)
        logger.info("[%s] Complete.", stage)

    # ------------------------------------------------------------------
    # Stage 7 — StoreIntoDatabases
    # ------------------------------------------------------------------
    async def _stage7_store_into_databases(
        self,
        entities: list[EntityNode],
        relations: list[EntityRelation],
        chunks: list[TextChunk],
    ) -> None:
        stage = "StoreIntoDatabases"
        if self._checker.is_stage_complete(stage):
            logger.info("[%s] Skipping — already complete.", stage)
            return

        logger.info("[%s] Writing to Neo4j + ES chunk index …", stage)

        graph: GraphStore = self._graph_factory.create(self._app_id)
        await graph.ensure_constraints()
        await graph.upsert_nodes(entities)
        await graph.upsert_edges(relations)

        # Embed + index chunks
        await self._indexer.index_chunks(chunks, self._app_id)

        self._checker.mark_stage_complete(stage)
        logger.info("[%s] Complete.", stage)

    # ------------------------------------------------------------------
    # Helpers — intermediate JSON
    # ------------------------------------------------------------------
    def _save_intermediate_json(self, data: list[dict], filename: str) -> None:
        path = self._dir / filename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)

    def _load_intermediate_json(self, filename: str) -> list[dict]:
        path = self._dir / filename
        if not path.exists():
            logger.warning("Intermediate file %s not found; returning empty.", filename)
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Helpers — KV loading (for Stage 1 skip)
    # ------------------------------------------------------------------
    async def _load_existing_chunks(self) -> list[TextChunk]:
        """Re-load all chunks for this app_id from KV when Stage 1 is skipped."""
        from autogen.models.storage import TextChunk as TC

        keys = await self._chunks_kv.keys()
        chunks: list[TC] = []
        for k in keys:
            chunk = await self._chunks_kv.get(k)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    async def _load_chunks_for_doc(self, doc_id: str) -> list[TextChunk]:
        """Load all chunks belonging to a specific document."""
        from autogen.models.storage import TextChunk

        # Brute-force scan — acceptable for moderate doc counts.
        # In production, add a secondary index (full_doc_id → chunk_ids).
        all_chunks = await self._load_existing_chunks()
        return [c for c in all_chunks if c.full_doc_id == doc_id]

    # ------------------------------------------------------------------
    # Name normalization application
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_name_map_to_entities(
        entities: list[EntityNode], name_map: dict[str, str]
    ) -> None:
        """Rewrite entity IDs and names according to the synonym map."""
        for e in entities:
            canonical = name_map.get(e.entity_name.lower())
            if canonical and canonical != e.entity_name:
                old_id = e.id
                old_name = e.entity_name
                e.entity_name = canonical
                # Rebuild ID from canonical name
                from autogen.extraction.extractor import EntityExtractor

                e.id = EntityExtractor._make_entity_id(canonical)  # noqa: SLF001
                logger.debug(
                    "Normalized entity '%s' → '%s'  (%s → %s)",
                    old_name,
                    canonical,
                    old_id,
                    e.id,
                )