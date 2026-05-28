"""File-backed key-value store — mirrors autogen.net LiteDB-backed KV storage.

Three namespaces per app_id (docs, chunks, strings) each stored as
individual JSON files under ``./kv/{namespace}/{key}.json``.

Temp-file-then-rename ensures crash-safe writes. Filter-before-upsert
enables idempotent re-ingestion. No network, no database — just files.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generic, TypeVar

import aiofiles
import aiofiles.os as aio_os

from autogen.logging.setup import get_logger

logger = get_logger("autogen.storage.file_kv")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Protocol shape — what factories return
# ---------------------------------------------------------------------------


class FileKvStorage(Generic[T]):
    """Generic file-backed key-value store for a single namespace.

    Each namespace becomes a subdirectory under ``./kv/``. Keys are
    stored as ``{key}.json`` files within that directory. The value
    type ``T`` must be serializable by ``json.dumps`` (Pydantic models
    work via ``.model_dump()``).

    Usage::

        store = FileKvStorage[FullDoc](namespace="neetpg_docs")
        doc = FullDoc(id="doc-abc", content="...")
        await store.upsert({"doc-abc": doc.model_dump()})
    """

    def __init__(self, namespace: str, base_dir: str = "./kv") -> None:
        """Create a KV store for the given namespace.

        Args:
            namespace: The namespace string, e.g. ``"neetpg_docs"``.
            base_dir: Root directory for all KV stores.
        """
        self._namespace = namespace
        self._dir = Path(base_dir) / namespace

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def namespace(self) -> str:
        return self._namespace

    async def filter_keys(self, keys: list[str]) -> list[str]:
        """Return the subset of *keys* that already exist in this store.

        Used to skip documents/chunks that were previously persisted,
        making re-ingestion idempotent.

        Args:
            keys: The keys to check.

        Returns:
            The subset of ``keys`` for which files already exist.
        """
        existing: list[str] = []
        for key in keys:
            path = self._path_for(key)
            if await aio_os.path.exists(path):
                existing.append(key)
        if existing:
            logger.debug(
                "file_kv.filter_keys.exists",
                namespace=self._namespace,
                count=len(existing),
            )
        return existing

    async def upsert(self, items: dict[str, object]) -> None:
        """Write or overwrite key-value pairs.

        Each item is serialized to JSON and written atomically using
        temp-file-then-rename. Serialization is handle by
        ``_serialize`` which calls ``model_dump(mode="json")`` on
        Pydantic models and falls back to ``dict`` / ``str``.

        Args:
            items: A mapping of key → value. Values can be Pydantic
                models, dicts, or JSON-serializable primitives.
        """
        if not items:
            return

        await self._ensure_dir()
        for key, value in items.items():
            path = self._path_for(key)
            serialized = self._serialize(value)
            await self._atomic_write(path, serialized)

        logger.info(
            "file_kv.upsert",
            namespace=self._namespace,
            count=len(items),
        )

    async def get(self, key: str) -> object | None:
        """Read a single value by key. Returns ``None`` if missing."""
        path = self._path_for(key)
        if not await aio_os.path.exists(path):
            return None
        async with aiofiles.open(path, encoding="utf-8") as f:
            raw = await f.read()
        return json.loads(raw)

    async def get_all(self, keys: list[str]) -> dict[str, object]:
        """Batch-read values. Missing keys are omitted from the result."""
        result: dict[str, object] = {}
        for key in keys:
            val = await self.get(key)
            if val is not None:
                result[key] = val
        return result

    async def delete(self, key: str) -> bool:
        """Delete a key. Returns ``True`` if the key existed."""
        path = self._path_for(key)
        existed = await aio_os.path.exists(path)
        if existed:
            await aio_os.remove(path)
            logger.debug(
                "file_kv.delete",
                namespace=self._namespace,
                key=key,
            )
        return existed

    async def keys(self) -> list[str]:
        """List all keys in this namespace."""
        await self._ensure_dir()
        try:
            files = await aio_os.listdir(str(self._dir))
        except FileNotFoundError:
            return []
        return [
            Path(f).stem for f in files if f.endswith(".json")
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, key: str) -> Path:
        # Sanitize key: replace path separators with underscores
        safe = key.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.json"

    async def _ensure_dir(self) -> None:
        await aio_os.makedirs(str(self._dir), exist_ok=True)

    @staticmethod
    def _serialize(value: object) -> str:
        """Convert a value to its JSON string representation.

        Pydantic models are serialized via ``model_dump(mode="json")``.
        Dicts and primitives are serialized directly.
        """
        # Pydantic v2 model
        if hasattr(value, "model_dump"):
            return json.dumps(
                value.model_dump(mode="json"),  # type: ignore[union-attr]
                ensure_ascii=False,
            )
        # Plain dict or primitive
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    async def _atomic_write(path: Path, content: str) -> None:
        """Write content atomically using temp-file-then-rename.

        On Windows, the rename may fail if the target exists, so we
        remove the target first.
        """
        # Write to a temp file in the same directory (ensures same filesystem)
        dir_path = path.parent
        await aio_os.makedirs(str(dir_path), exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix=f"{path.stem}_", dir=str(dir_path)
        )
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            # Atomic rename on POSIX; on Windows, replace if exists
            if os.name == "nt" and path.exists():
                path.unlink()
            Path(tmp_path).rename(path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Factory — mirrors autogen.net IKeyValueStorageFactory
# ---------------------------------------------------------------------------


class FileKvStorageFactory:
    """Factory that creates FileKvStorage instances per namespace.

    Mirrors autogen.net RegisterServices.cs:116-118 pattern::

        factory.Create[FullDoc]("{appId}_docs")
        factory.Create[TextChunk]("{appId}_chunks")
        factory.Create[str]("{appId}_strings")

    Usage::

        factory = FileKvStorageFactory(base_dir="./kv")
        docs_store = factory.create("neetpg_docs")
        await docs_store.upsert({"doc-abc": full_doc.model_dump()})
    """

    def __init__(self, base_dir: str = "./kv") -> None:
        self._base_dir = base_dir

    @property
    def base_dir(self) -> str:
        return self._base_dir

    def create(self, namespace: str) -> FileKvStorage[object]:
        """Create a FileKvStorage for the given namespace.

        Args:
            namespace: e.g. ``"neetpg_docs"``, ``"neetpg_chunks"``, ``"neetpg_strings"``.

        Returns:
            A new FileKvStorage instance bound to that namespace.
        """
        return FileKvStorage[object](namespace=namespace, base_dir=self._base_dir)