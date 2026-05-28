"""Tests for TextChunker (class) and chunk_text (module function) — Phase 3."""

from __future__ import annotations

import pytest

from autogen.chunking.chunker import TextChunker, chunk_text, md5_hash_id
from autogen.models.storage import TextChunk


@pytest.fixture
def chunker() -> TextChunker:
    return TextChunker(
        chunk_token_size=20,
        chunk_overlap_token_size=5,
        tiktoken_model_name="cl100k_base",
    )


class TestEmptyAndShort:
    def test_empty_content_returns_no_chunks(self, chunker):
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_short_content_produces_one_chunk(self, chunker):
        chunks = chunker.chunk_text("Hello world.")
        assert len(chunks) == 1

    def test_chunk_carries_metadata(self, chunker):
        chunks = chunker.chunk_text("Hello world.", app_id="neetpg", full_doc_id="doc-001")
        assert chunks[0].full_doc_id == "doc-001"
        assert chunks[0].app_id == "neetpg"

    def test_module_function_empty_returns_empty(self):
        assert chunk_text("") == []


class TestChunkIds:
    def test_chunk_id_is_deterministic(self, chunker):
        chunks_a = chunker.chunk_text("Same content every time.")
        chunks_b = chunker.chunk_text("Same content every time.")
        assert [c.id for c in chunks_a] == [c.id for c in chunks_b]

    def test_chunk_ids_unique_within_doc(self, chunker):
        long_text = " ".join([f"word{i}" for i in range(200)])
        chunks = chunker.chunk_text(long_text)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_order_sequential(self, chunker):
        long_text = " ".join(["word"] * 200)
        chunks = chunker.chunk_text(long_text)
        assert [c.order for c in chunks] == list(range(len(chunks)))


class TestTokenCounts:
    def test_tokens_count_populated(self, chunker):
        chunks = chunker.chunk_text("Hello world. This is a test document.")
        for chunk in chunks:
            assert chunk.tokens_count > 0

    def test_first_chunk_within_size(self, chunker):
        long_text = " ".join(["word"] * 200)
        chunks = chunker.chunk_text(long_text)
        assert chunks[0].tokens_count <= 20

    def test_count_tokens_method(self, chunker):
        n = chunker.count_tokens("hello world")
        assert n > 0


class TestOverlap:
    def test_multiple_chunks_on_long_text(self, chunker):
        long_text = " ".join(["token"] * 100)
        chunks = chunker.chunk_text(long_text)
        assert len(chunks) > 1

    def test_consecutive_chunks_share_words(self, chunker):
        """Overlap means the end of chunk[0] and start of chunk[1] share tokens."""
        long_text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunker.chunk_text(long_text)
        assert len(chunks) >= 2
        words_0 = set(chunks[0].content.split())
        words_1 = set(chunks[1].content.split())
        assert words_0 & words_1, "expected overlap between consecutive chunks"


class TestTruncation:
    def test_truncate_to_tokens(self, chunker):
        text = " ".join(["word"] * 100)
        truncated = chunker.truncate_to_tokens(text, 10)
        assert chunker.count_tokens(truncated) <= 10

    def test_truncate_noop_when_short(self, chunker):
        assert chunker.truncate_to_tokens("hello", 100) == "hello"


class TestMd5HashId:
    def test_deterministic(self):
        assert md5_hash_id("content", "chunk") == md5_hash_id("content", "chunk")

    def test_prefix_in_result(self):
        result = md5_hash_id("content", "myprefix")
        assert result.startswith("myprefix-")

    def test_different_content_different_id(self):
        assert md5_hash_id("aaa", "chunk") != md5_hash_id("bbb", "chunk")

    def test_method_matches_function(self, chunker):
        assert chunker.md5_hash_id("x", "p") == md5_hash_id("x", "p")


class TestDefaultSettings:
    def test_default_chunker_uses_960_tokens(self):
        chunker = TextChunker()
        assert chunker._chunk_size == 960
        assert chunker._overlap == 128
