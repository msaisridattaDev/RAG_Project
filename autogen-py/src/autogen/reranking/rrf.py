"""combine_results — Reciprocal Rank Fusion (RRF).

Merges multiple ranked lists into one using rank-only scoring.
Because raw scores from different searches aren't comparable (one might use
cosine similarity 0-1, another might use BM25 0-∞), but ranks always are.

Formula:
    Each item's RRF score = sum(1 / (k + rank_in_list)) across all lists
    it appears in.

The constant k (default 60) smooths the function so early ranks don't
dominate too heavily.

This mirrors the Agentic.Memory.Storage.Utils.CombineResults in the .NET source.
"""

from __future__ import annotations

from collections import defaultdict


def combine_results(
    per_keyword: list[list[tuple[str, float]]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each input list contains (id, score) pairs sorted by score descending.
    The function ignores the raw scores and uses only the rank position.

    Args:
        per_keyword: A list of ranked lists. Each inner list contains
            (id, score) tuples sorted by score descending.
            The score values are ignored; only rank matters.
        k: The RRF constant (default 60). Higher values produce more
            graduated scores. Standard range: 1-100.
        top_n: If set, return only the top N results. If None, return all.

    Returns:
        A list of (id, rrf_score) tuples sorted by rrf_score descending.
        The RRF score is the sum of 1/(k + rank) across all lists.

    Examples:
        >>> combine_results([
        ...     [("A", 0.9), ("B", 0.8)],
        ...     [("B", 0.9), ("A", 0.8)],
        ... ])
        [('A', 0.032...), ('B', 0.032...)]

        >>> combine_results([
        ...     [("A", 0.9)],
        ...     [("B", 0.9)],
        ... ])
        [('A', 0.016...), ('B', 0.016...)]
    """
    if not per_keyword:
        return []

    # Accumulate RRF scores per item
    rrf_scores: dict[str, float] = defaultdict(float)

    for ranked_list in per_keyword:
        for rank, (item_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[item_id] += 1.0 / (k + rank)

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    return sorted_items
