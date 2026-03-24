"""
Evaluation framework for the recommendation system.

Metrics:
  - Rating prediction: RMSE, MAE
  - Ranking quality:   Precision@K, Recall@K, NDCG@K
  - Coverage:          Catalog coverage
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rating prediction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root Mean Squared Error."""
    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Absolute Error."""
    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)
    return float(np.mean(np.abs(arr_true - arr_pred)))


def evaluate_predictions(test_df: pd.DataFrame) -> dict[str, float]:
    """
    Evaluate rating predictions from a DataFrame.

    Args:
        test_df: DataFrame with columns [rating, predicted_rating].

    Returns:
        Dict with rmse and mae.
    """
    y_true = test_df["rating"].tolist()
    y_pred = test_df["predicted_rating"].tolist()
    return {
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def _dcg(relevances: list[float]) -> float:
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances)
    )


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @K.

    Args:
        recommended: Ordered list of recommended movie IDs.
        relevant: Set of ground-truth relevant movie IDs.
        k: Cutoff.

    Returns:
        NDCG@K score in [0, 1].
    """
    top_k = recommended[:k]
    gains = [1.0 if mid in relevant else 0.0 for mid in top_k]
    dcg = _dcg(gains)
    ideal = _dcg([1.0] * min(len(relevant), k))
    return float(dcg / ideal) if ideal > 0 else 0.0


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended[:k]
    hits = sum(1 for mid in top_k if mid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items found in the top-K."""
    top_k = recommended[:k]
    hits = sum(1 for mid in top_k if mid in relevant)
    return hits / len(relevant) if relevant else 0.0


# ---------------------------------------------------------------------------
# Batch evaluation across users
# ---------------------------------------------------------------------------

def evaluate_ranking(
    recommender,
    test_df: pd.DataFrame,
    all_movie_ids: list[int],
    k: int = 10,
    relevance_threshold: float = 4.0,
    n_users: int = 500,
    sample_candidates: int = 1000,
) -> dict[str, float]:
    """
    Evaluate a recommender's ranking quality on a test set.

    For each sampled user:
      1. Build the ground-truth set (movies rated >= relevance_threshold in test).
      2. Sample candidate movies (ground-truth + random negatives).
      3. Get recommendations.
      4. Compute Precision@K, Recall@K, NDCG@K.

    Args:
        recommender: Any object with a .recommend(user_id, candidate_ids, top_n) method.
        test_df: DataFrame with [user_id, movie_id, rating].
        all_movie_ids: Full list of movie IDs in the catalog.
        k: Cutoff rank.
        relevance_threshold: Min rating to count as relevant.
        n_users: Max number of users to evaluate (for speed).
        sample_candidates: Candidate pool size per user.

    Returns:
        Dict with precision@k, recall@k, ndcg@k (averaged across users).
    """
    rng = np.random.default_rng(42)

    # Group test ratings by user
    user_relevant: dict[int, set[int]] = defaultdict(set)
    for _, row in test_df.iterrows():
        if row["rating"] >= relevance_threshold:
            user_relevant[int(row["user_id"])].add(int(row["movie_id"]))

    eligible_users = [uid for uid, rel in user_relevant.items() if rel]
    sampled_users = eligible_users[:n_users]

    all_ids_arr = np.array(all_movie_ids)
    precisions, recalls, ndcgs = [], [], []

    for user_id in sampled_users:
        relevant = user_relevant[user_id]

        # Candidate pool: all relevant + random negatives
        negatives = rng.choice(
            all_ids_arr,
            size=min(sample_candidates - len(relevant), len(all_ids_arr)),
            replace=False,
        ).tolist()
        candidates = list(relevant) + negatives

        try:
            recs = recommender.recommend(user_id, candidates, top_n=k)
            rec_ids = [r["movie_id"] for r in recs]
        except Exception:
            continue

        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))

    if not precisions:
        return {"precision@k": 0.0, "recall@k": 0.0, "ndcg@k": 0.0, "k": k, "n_users": 0}

    return {
        f"precision@{k}": round(float(np.mean(precisions)), 4),
        f"recall@{k}": round(float(np.mean(recalls)), 4),
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 4),
        "k": k,
        "n_users": len(precisions),
    }


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def catalog_coverage(
    recommendations: list[list[int]], all_movie_ids: list[int]
) -> float:
    """
    Fraction of the catalog that appears in at least one recommendation list.

    Args:
        recommendations: List of recommendation lists (each is a list of movie IDs).
        all_movie_ids: Full catalog.

    Returns:
        Coverage ratio in [0, 1].
    """
    recommended_set = {mid for recs in recommendations for mid in recs}
    return len(recommended_set) / len(all_movie_ids) if all_movie_ids else 0.0
