"""Seed a TextChunk index with a small CSV of test data.

Usage:
    uv run python scripts/seed_chunks.py --app-id neetpg --csv scripts/data/medical_seed.csv

CSV format (header required):
    id,content
    chunk-001,"Aspirin inhibits cyclooxygenase, reducing prostaglandin synthesis ..."
    chunk-002,"Inflammation involves edema, erythema, heat, and pain ..."

Steps performed:
    1. Load Settings (reads .env).
    2. Build JinaEmbeddingClient + AsyncElasticsearch client.
    3. Build VectorStoreFactory.create(app_id, TextChunk) → textchunk_{app_id}_1024.
    4. Read CSV rows, embed contents (retrieval.passage), upsert in bulk.
    5. Print a summary and exit.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

# Make src/ importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autogen.config.settings import Settings
from autogen.embeddings.jina import JinaEmbeddingClient
from autogen.models.storage import TextChunk
from autogen.storage.elastic import VectorStoreFactory, _create_es_client


async def _seed(app_id: str, csv_path: Path) -> int:
    settings = Settings()

    if app_id not in settings.app_identity.allowed_app_ids:
        print(
            f"ERROR: app_id={app_id!r} not in APP_IDENTITY__ALLOWED_APP_IDS="
            f"{settings.app_identity.allowed_app_ids}"
        )
        return 2

    if not settings.embedding_options.api_key:
        print("ERROR: JINA_API_KEY is not set in environment or .env")
        return 2

    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("id") or not r.get("content"):
                continue
            rows.append({"id": r["id"].strip(), "content": r["content"].strip()})

    if not rows:
        print(f"ERROR: no usable rows in {csv_path}")
        return 2

    print(f"Loaded {len(rows)} rows from {csv_path}")

    embedding_client = JinaEmbeddingClient(settings.embedding_options)
    es_client = _create_es_client(settings)
    factory = VectorStoreFactory(es_client, embedding_client, dim=settings.elasticsearch.embedding_dim)
    store = factory.create(app_id, TextChunk)

    try:
        await store.ensure_index()

        contents = [r["content"] for r in rows]
        vectors = await embedding_client.embed(contents, task="retrieval.passage")
        if len(vectors) != len(rows):
            print(f"ERROR: embed returned {len(vectors)} vectors for {len(rows)} rows")
            return 3

        chunks = [
            TextChunk(
                id=r["id"],
                content=r["content"],
                embedding=v,
                app_id=app_id,
                tokens_count=0,
                full_doc_id="seed",
                order=i,
            )
            for i, (r, v) in enumerate(zip(rows, vectors))
        ]

        upserted = await store.upsert(chunks)
        print(f"OK: upserted {upserted} chunks into index {store.index_name}")
        return 0
    finally:
        await embedding_client.close()
        await es_client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed a TextChunk index for an app_id.")
    parser.add_argument("--app-id", required=True, help="Tenant ID (e.g. neetpg)")
    parser.add_argument(
        "--csv",
        default="scripts/data/medical_seed.csv",
        help="Path to CSV with id,content columns",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        return 2

    return asyncio.run(_seed(args.app_id, csv_path))


if __name__ == "__main__":
    raise SystemExit(main())
