"""Document chunker — mirrors autogen.net chunk_text and md5_hash_id.

Splits documents into overlapping windows of 960 tokens with 128-token
overlap using tiktoken's cl100k_base encoding. Generates deterministic
MD5-based IDs for idempotent re-ingestion.

Config values from LightRagConfig.cs:15-17 and EntityExtractionPipeline.cs:135.
"""

from __future__ import annotations

import hashlib
from typing import Any

import tiktoken

from autogen.config.settings import LightRagSettings
from autogen.logging.setup import get_logger
from autogen.models.storage import TextChunk

logger = get_logger("autogen.chunking")


def md5_hash_id(content: str, prefix: str) -> str:
    """Generate a deterministic, content-addressed ID.

    Same content always produces the same ID. The prefix argument
    namespaces the ID (e.g. ``"doc"``, ``"chunk"``) so you can tell
    them apart in logs and KV stores.

    MD5 is fine here — we care about a stable, fast, well-distributed
    hash, not cryptographic security.

    Args:
        content: The text to hash.
        prefix: A string prefix for the ID, e.g. ``"doc"`` or ``"chunk"``.

    Returns:
        A string like ``"chunk-3f9a2b1c..."`` (20 hex chars after the dash).
    """
    h = hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]
    return f"{prefix}-{h}"


def chunk_text(
    content: str,
    chunk_tokens: int = 960,
    overlap_tokens: int = 128,
    model: str = "cl100k_base",
    *,
    app_id: str = "neetpg",
    full_doc_id: str = "",
) -> list[TextChunk]:
    """Split a document into overlapping token-window chunks.

    Encodes the full content with tiktoken using the given encoding
    name, slides a window of ``chunk_tokens`` tokens with a step of
    ``chunk_tokens - overlap_tokens``, decodes each window back to
    text, and returns a list of ``TextChunk`` objects.

    The returned chunks have stable, content-addressed IDs so
    re-chunking the same text produces identical IDs.

    Args:
        content: The raw document text.
        chunk_tokens: Window size in tokens (default 960, matches LightRagConfig.cs:15).
        overlap_tokens: Overlap between adjacent windows in tokens (default 128, matches LightRagConfig.cs:16).
        model: Tiktoken encoding name (default ``"cl100k_base"``, matches LightRagConfig.cs:17).
        app_id: Tenant scope for the chunks.
        full_doc_id: The ``FullDoc.id`` this chunk belongs to.

    Returns:
        A list of ``TextChunk`` objects, one per window, in document order.
    """
    if not content:
        logger.warning("chunk.empty_input", app_id=app_id)
        return []

    from autogen.config.settings import Settings

    settings = Settings()
    settings_chunk = settings.lightrag.chunk_token_size
    settings_overlap = settings.lightrag.chunk_overlap_token_size
    encoding_name = settings.lightrag.tiktoken_model_name

    # Use explicit args first, fall back to settings
    chunk_tokens = chunk_tokens if chunk_tokens != 960 else settings_chunk
    overlap_tokens = overlap_tokens if overlap_tokens != 128 else settings_overlap
    model = model if model != "cl100k_base" else encoding_name

    try:
        enc = tiktoken.get_encoding(model)
    except Exception:
        logger.error("chunk.tiktoken_failed", encoding=model)
        raise

    tokens: list[int] = enc.encode(content)
    total_tokens = len(tokens)

    if total_tokens == 0:
        logger.warning("chunk.zero_tokens_after_encode", app_id=app_id)
        return []

    step = chunk_tokens - overlap_tokens
    if step <= 0:
        raise ValueError(
            f"Overlap ({overlap_tokens}) must be less than chunk size ({chunk_tokens})"
        )

    chunks: list[TextChunk] = []

    # Slide window
    start = 0
    order = 0
    while start < total_tokens:
        end = min(start + chunk_tokens, total_tokens)
        window: list[int] = tokens[start:end]
        text = enc.decode(window)

        chunk_id = md5_hash_id(text, "chunk")
        chunks.append(
            TextChunk(
                id=chunk_id,
                content=text,
                full_doc_id=full_doc_id,
                order=order,
                tokens_count=len(window),
                app_id=app_id,
            )
        )

        order += 1
        if end >= total_tokens:
            break
        start += step

    logger.info(
        "chunk.complete",
        total_tokens=total_tokens,
        chunk_count=len(chunks),
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        app_id=app_id,
    )

    return chunks


class TextChunker:
    """Class wrapper around chunk_text() for pipeline and LightRag injection.

    Holds chunk/overlap/encoding config so callers can construct once and
    call .chunk_text() per document, matching the chunker=... constructor arg
    in EntityExtractionPipeline and LightRag.build().
    """

    def __init__(
        self,
        chunk_token_size: int = 960,
        chunk_overlap_token_size: int = 128,
        tiktoken_model_name: str = "cl100k_base",
    ) -> None:
        self._chunk_size = chunk_token_size
        self._overlap = chunk_overlap_token_size
        self._model = tiktoken_model_name
        try:
            self._enc = tiktoken.get_encoding(tiktoken_model_name)
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk_text(
        self,
        content: str,
        app_id: str = "neetpg",
        full_doc_id: str = "",
    ) -> list[TextChunk]:
        """Split content into overlapping TextChunk windows."""
        return chunk_text(
            content,
            chunk_tokens=self._chunk_size,
            overlap_tokens=self._overlap,
            model=self._model,
            app_id=app_id,
            full_doc_id=full_doc_id,
        )

    def chunk(
        self,
        content: str,
        app_id: str = "neetpg",
        full_doc_id: str = "",
    ) -> list[TextChunk]:
        """Alias for chunk_text() kept for backward compatibility."""
        return self.chunk_text(content, app_id=app_id, full_doc_id=full_doc_id)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in ``text`` using this chunker's encoding."""
        return len(self._enc.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate ``text`` to at most ``max_tokens`` tokens, decoded back to str."""
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._enc.decode(tokens[:max_tokens])

    @staticmethod
    def md5_hash_id(content: str, prefix: str) -> str:
        """Delegate to the module-level md5_hash_id()."""
        return md5_hash_id(content, prefix)