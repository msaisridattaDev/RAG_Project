"""Tests for the reranking module (Day 5 primitives).

Tests cover:
    - RerankClient (with mock HTTP server)
    - find_elbow_index (pure function, no external deps)
    - combine_results / RRF (pure function, no external deps)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.reranking import RerankClient, combine_results, find_elbow_index

# ======================================================================
# RerankClient tests
# ======================================================================


class TestRerankClient:
    """Tests for RerankClient using a mocked httpx client."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock httpx.AsyncClient."""
        client = MagicMock(spec=AsyncMock)
        client.post = AsyncMock()
        return client

    @pytest.fixture
    def reranker(self, mock_client: MagicMock) -> RerankClient:
        """Create a RerankClient with a mocked HTTP client."""
        return RerankClient(
            base_url="http://test-reranker:8077",
            api_key=None,
            model="Qwen/Qwen3-Reranker-4B",
            http_client=mock_client,  # type: ignore[arg-type]
        )

    async def test_rerank_basic(self, reranker: RerankClient, mock_client: MagicMock) -> None:
        """Test basic rerank with two documents."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95, "document": "doc2 text"},
                {"index": 0, "relevance_score": 0.85, "document": "doc1 text"},
            ]
        }
        mock_client.post.return_value = mock_response

        hits = await reranker.rerank(
            query="test query",
            documents=["doc1 text", "doc2 text"],
            top_k=2,
        )

        assert len(hits) == 2
        assert hits[0].index == 1
        assert hits[0].relevance_score == 0.95
        assert hits[0].document == "doc2 text"
        assert hits[1].index == 0
        assert hits[1].relevance_score == 0.85

        # Verify the request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://test-reranker:8077/rerank"
        assert call_args[1]["json"] == {
            "model": "Qwen/Qwen3-Reranker-4B",
            "query": "test query",
            "documents": ["doc1 text", "doc2 text"],
            "top_n": 2,
        }

    async def test_rerank_empty_documents(self, reranker: RerankClient, mock_client: MagicMock) -> None:
        """Test rerank with empty documents list."""
        hits = await reranker.rerank(query="test", documents=[])
        assert hits == []
        mock_client.post.assert_not_called()

    async def test_rerank_with_api_key(self) -> None:
        """Test that API key is sent in Authorization header."""
        mock_client = MagicMock(spec=AsyncMock)
        mock_client.post = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_client.post.return_value = mock_response

        reranker = RerankClient(
            base_url="http://test:8077",
            api_key="sk-test-key",
            http_client=mock_client,  # type: ignore[arg-type]
        )

        await reranker.rerank(query="test", documents=["doc1"])

        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-key"

    async def test_rerank_no_top_k(self, reranker: RerankClient, mock_client: MagicMock) -> None:
        """Test rerank without top_k (should return all)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9, "document": "doc1"},
                {"index": 1, "relevance_score": 0.8, "document": "doc2"},
            ]
        }
        mock_client.post.return_value = mock_response

        hits = await reranker.rerank(query="test", documents=["doc1", "doc2"])

        assert len(hits) == 2
        # Verify top_n is not in the request body
        call_kwargs = mock_client.post.call_args[1]
        assert "top_n" not in call_kwargs["json"]

    async def test_rerank_http_error(self, reranker: RerankClient, mock_client: MagicMock) -> None:
        """Test that HTTP errors propagate."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )
        mock_client.post.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            await reranker.rerank(query="test", documents=["doc1"])

    async def test_rerank_close(self, reranker: RerankClient, mock_client: MagicMock) -> None:
        """Test that close() calls the underlying client's close."""
        mock_client.aclose = AsyncMock()
        await reranker.close()
        mock_client.aclose.assert_called_once()


# ======================================================================
# find_elbow_index tests
# ======================================================================


class TestFindElbowIndex:
    """Tests for the elbow cutoff function."""

    def test_clear_elbow(self) -> None:
        """Scores with a clear drop should return the index before the drop."""
        scores = [0.92, 0.91, 0.90, 0.42, 0.41, 0.40]
        assert find_elbow_index(scores) == 2

    def test_no_clear_elbow(self) -> None:
        """Smooth scores should return the last index (take all)."""
        scores = [0.62, 0.61, 0.61, 0.60, 0.59, 0.58]
        assert find_elbow_index(scores) == 5

    def test_flat_scores(self) -> None:
        """All equal scores should return the last index."""
        scores = [0.65, 0.65, 0.64, 0.63]
        assert find_elbow_index(scores) == 3

    def test_single_score(self) -> None:
        """Single score should return 0."""
        scores = [0.85]
        assert find_elbow_index(scores) == 0

    def test_empty_list(self) -> None:
        """Empty list should return -1."""
        scores: list[float] = []
        assert find_elbow_index(scores) == -1

    def test_two_scores_with_gap(self) -> None:
        """Two scores with a gap should return 0 (take the first)."""
        scores = [0.95, 0.50]
        assert find_elbow_index(scores) == 0

    def test_two_scores_close(self) -> None:
        """Two close scores should return 1 (take both)."""
        scores = [0.62, 0.61]
        assert find_elbow_index(scores) == 1

    def test_large_gap_in_middle(self) -> None:
        """Large gap in the middle should return the index before it."""
        scores = [0.9, 0.8, 0.7, 0.3, 0.29, 0.28]
        assert find_elbow_index(scores) == 2

    def test_very_small_gap(self) -> None:
        """Very small gaps should be treated as flat."""
        scores = [0.501, 0.500, 0.499, 0.498]
        assert find_elbow_index(scores) == 3


# ======================================================================
# combine_results (RRF) tests
# ======================================================================


class TestCombineResults:
    """Tests for Reciprocal Rank Fusion."""

    def test_single_list(self) -> None:
        """Single list should be returned as-is (just sorted)."""
        result = combine_results([
            [("A", 0.9), ("B", 0.8), ("C", 0.7)],
        ])
        assert len(result) == 3
        assert result[0][0] == "A"
        assert result[1][0] == "B"
        assert result[2][0] == "C"

    def test_two_lists_same_items(self) -> None:
        """Two lists with same items should boost items high in both."""
        result = combine_results([
            [("A", 0.9), ("B", 0.8)],
            [("B", 0.9), ("A", 0.8)],
        ])
        # Both A and B appear at rank 1 in one list and rank 2 in the other
        # A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.0325
        # B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.0325
        assert len(result) == 2
        # Scores should be approximately equal
        assert abs(result[0][1] - result[1][1]) < 0.001

    def test_two_lists_different_items(self) -> None:
        """Items appearing in only one list should rank lower."""
        result = combine_results([
            [("A", 0.9)],
            [("B", 0.9)],
        ])
        # A: 1/(60+1) ≈ 0.0164
        # B: 1/(60+1) ≈ 0.0164
        assert len(result) == 2
        assert abs(result[0][1] - result[1][1]) < 0.001

    def test_item_in_multiple_lists_boosted(self) -> None:
        """Item appearing in multiple lists should be boosted."""
        result = combine_results([
            [("A", 0.9), ("B", 0.8)],
            [("A", 0.9), ("C", 0.8)],
            [("A", 0.9), ("D", 0.8)],
        ])
        # A appears at rank 1 in all 3 lists → highest score
        assert result[0][0] == "A"
        # B, C, D each appear at rank 2 in one list → lower scores
        assert result[1][0] in ("B", "C", "D")

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        result = combine_results([])
        assert result == []

    def test_empty_lists(self) -> None:
        """Lists with empty inner lists should be handled."""
        result = combine_results([
            [],
            [],
        ])
        assert result == []

    def test_top_n(self) -> None:
        """top_n parameter should limit results."""
        result = combine_results([
            [("A", 0.9), ("B", 0.8), ("C", 0.7)],
            [("B", 0.9), ("C", 0.8), ("A", 0.7)],
        ], top_n=2)
        assert len(result) == 2

    def test_custom_k(self) -> None:
        """Custom k value should affect scores."""
        result_k1 = combine_results([
            [("A", 0.9), ("B", 0.8)],
        ], k=1)
        result_k60 = combine_results([
            [("A", 0.9), ("B", 0.8)],
        ], k=60)

        # With k=1, rank 1 gets 1/2 = 0.5, rank 2 gets 1/3 ≈ 0.333
        # With k=60, rank 1 gets 1/61 ≈ 0.0164, rank 2 gets 1/62 ≈ 0.0161
        assert result_k1[0][1] > result_k60[0][1]
        # The ratio between rank 1 and rank 2 should be different
        ratio_k1 = result_k1[0][1] / result_k1[1][1]
        ratio_k60 = result_k60[0][1] / result_k60[1][1]
        assert ratio_k1 > ratio_k60  # k=1 is more aggressive
