"""Evaluation module: recall@k metric for retrieval quality."""

from __future__ import annotations

from typing import Dict, List

from src.models import MinimalSource

_OVERLAP_THRESHOLD = 0.05


def _overlap_ratio(correct: MinimalSource, retrieved: MinimalSource) -> float:
    """Fraction of `correct`'s character span covered by `retrieved`."""
    if correct.file_path != retrieved.file_path:
        return 0.0

    start = max(correct.first_character_index, retrieved.first_character_index)
    end = min(correct.last_character_index, retrieved.last_character_index)
    overlap = max(0, end - start)

    correct_len = max(1, correct.last_character_index - correct.first_character_index)
    return overlap / correct_len


def is_found(
    correct: MinimalSource,
    retrieved_sources: List[MinimalSource],
    threshold: float = _OVERLAP_THRESHOLD,
) -> bool:
    """A correct source counts as found if any retrieved source overlaps it by >= threshold."""
    return any(_overlap_ratio(correct, r) >= threshold for r in retrieved_sources)


def recall_at_k(
    correct_sources: List[MinimalSource],
    retrieved_sources: List[MinimalSource],
    k: int,
    threshold: float = _OVERLAP_THRESHOLD,
) -> float:
    """recall@k = number_found / total_number_of_correct_sources, over the top-k retrieved."""
    if not correct_sources:
        return 1.0

    top_k = retrieved_sources[:k]
    found = sum(1 for c in correct_sources if is_found(c, top_k, threshold))
    return found / len(correct_sources)


def evaluate(
    ground_truth: Dict[str, List[MinimalSource]],
    predictions: Dict[str, List[MinimalSource]],
    ks: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute average recall@k for each k, over questions present in both dicts.

    Args:
        ground_truth: question_id -> list of correct MinimalSource.
        predictions: question_id -> list of retrieved MinimalSource (ranked).
        ks: list of k values to evaluate.

    Returns:
        Dict mapping "recall@k" -> average score.
    """
    shared_ids = [qid for qid in ground_truth if qid in predictions]

    results: Dict[str, float] = {}
    for k in ks:
        if not shared_ids:
            results[f"recall@{k}"] = 0.0
            continue
        scores = [
            recall_at_k(ground_truth[qid], predictions[qid], k)
            for qid in shared_ids
        ]
        results[f"recall@{k}"] = sum(scores) / len(scores)

    return results
