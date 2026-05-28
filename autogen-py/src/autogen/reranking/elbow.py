"""find_elbow_index — adaptive top-K via elbow detection.

Given a descending-sorted list of scores, finds the "elbow" — the natural
cutoff point where scores drop sharply. Everything after the elbow is noise.

Algorithm: maximum-gap variant.
    For sorted scores s[0], s[1], ..., s[n-1], compute gaps s[i] - s[i+1].
    The elbow is at the index before the largest gap.

This mirrors the Agentic.Memory.Storage.Utils.FindElbowIndex in the .NET source.

Edge cases:
    - All scores equal (flat) → return last index (take all).
    - Only one score → return 0.
    - Empty list → return -1 (caller's responsibility to handle).
"""

from __future__ import annotations


def find_elbow_index(scores: list[float]) -> int:
    """Find the elbow index in a descending-sorted list of scores.

    The elbow is the point where the score drops most sharply. Everything
    at or before the elbow is considered a "real" result; everything after
    is noise.

    Args:
        scores: A list of scores sorted in descending order.
            Must not be empty.

    Returns:
        The index of the elbow (inclusive). The caller should take
        results[:elbow_index + 1].

        Examples:
            [0.92, 0.91, 0.90, 0.42, 0.41, 0.40] → 2
            [0.62, 0.61, 0.61, 0.60, 0.59, 0.58] → 5 (no clear elbow)
            [0.85] → 0
            [] → -1

    Notes:
        The algorithm uses a relative-gap heuristic:
        1. Compute gaps between consecutive scores.
        2. Find the largest gap.
        3. The elbow is at the index before the largest gap, BUT only if
           that gap is significantly larger than the average gap.
        4. If no gap is significantly larger than average, return the last
           index (take all — no clear elbow).
    """
    n = len(scores)

    if n == 0:
        return -1
    if n == 1:
        return 0

    # Compute gaps between consecutive scores
    gaps: list[float] = []
    for i in range(n - 1):
        gaps.append(scores[i] - scores[i + 1])

    # Find the largest gap
    max_gap = gaps[0]
    max_gap_idx = 0

    for i in range(1, len(gaps)):
        if gaps[i] > max_gap:
            max_gap = gaps[i]
            max_gap_idx = i

    # If all gaps are zero (perfectly flat), take all
    if max_gap == 0:
        return n - 1

    # Compute the average gap (excluding the max gap to avoid self-bias)
    other_gaps_sum = sum(gaps) - max_gap
    other_gaps_count = len(gaps) - 1
    avg_other_gap = other_gaps_sum / other_gaps_count if other_gaps_count > 0 else 0

    # For exactly 2 scores (only one gap), the single gap is always the max.
    # If the gap is large enough relative to the score magnitude, it's an elbow.
    if other_gaps_count == 0:
        # Only one gap exists. If the gap is > 10% of the first score, it's an elbow.
        if scores[0] > 0 and max_gap / scores[0] > 0.1:
            return max_gap_idx
        return n - 1

    # The elbow is significant only if the max gap is at least 3x the
    # average of the other gaps. This handles:
    #   - Clear elbows: max_gap >> avg_other_gap → return elbow
    #   - Smooth distributions: max_gap ≈ avg_other_gap → return last
    if max_gap >= 3.0 * avg_other_gap:
        return max_gap_idx

    # No clear elbow — take all
    return n - 1



