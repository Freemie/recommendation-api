"""
Hybrid Recommender — blends collaborative + content-based scores.

Strategy:
  - Known user with ratings  → weighted blend (CF-heavy)
  - Cold-start user           → content-based only (genre fallback)
  - Unknown movie             → falls back to content-based similarity

Score formula:
    hybrid_score = α * cf_score_norm + (1 - α) * cb_score
    where α = cf_weight (default 0.7)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .collaborative import CollaborativeFilter
from .content_based import ContentBasedFilter

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Rating scale bounds used for min-max normalisation of CF predictions
RATING_MIN, RATING_MAX = 0.5, 5.0


def _normalize_cf_score(score: float) -> float:
    return (score - RATING_MIN) / (RATING_MAX - RATING_MIN)


class HybridRecommender:
    """
    Combines CollaborativeFilter and ContentBasedFilter.

    Usage:
        hybrid = HybridRecommender(cf_model, cb_model, cf_weight=0.7)
        recs = hybrid.recommend(user_id=42, candidate_ids=[...], top_n=10)
    """

    def __init__(
        self,
        cf_model: CollaborativeFilter,
        cb_model: ContentBasedFilter,
        cf_weight: float = 0.7,
        cold_start_min_ratings: int = 5,
    ):
        if not (0.0 <= cf_weight <= 1.0):
            raise ValueError("cf_weight must be between 0 and 1.")
        self.cf = cf_model
        self.cb = cb_model
        self.cf_weight = cf_weight
        self.cold_start_min_ratings = cold_start_min_ratings

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        candidate_movie_ids: list[int],
        top_n: int = 10,
        already_rated: set[int] | None = None,
        user_rated_movies: list[int] | None = None,
    ) -> list[dict]:
        """
        Return top-N recommendations for a user.

        Args:
            user_id: Target user.
            candidate_movie_ids: Pool of movies to rank.
            top_n: Number of results to return.
            already_rated: Movie IDs to exclude (already seen).
            user_rated_movies: Movies the user has rated (used for CB profile).

        Returns:
            List of dicts: {movie_id, hybrid_score, cf_score, cb_score, strategy}
        """
        exclude = already_rated or set()
        candidates = [mid for mid in candidate_movie_ids if mid not in exclude]

        is_cold_start = (
            self.cf.get_user_embedding(user_id) is None
            or len(user_rated_movies or []) < self.cold_start_min_ratings
        )

        if is_cold_start:
            return self._cold_start_recommend(user_id, candidates, top_n)

        return self._blend_recommend(user_id, candidates, top_n, user_rated_movies or [])

    def _blend_recommend(
        self,
        user_id: int,
        candidates: list[int],
        top_n: int,
        user_rated_movies: list[int],
    ) -> list[dict]:
        # CF scores
        cf_raw: dict[int, float] = {}
        for mid in candidates:
            try:
                cf_raw[mid] = self.cf.predict(user_id, mid)
            except Exception:
                cf_raw[mid] = (RATING_MIN + RATING_MAX) / 2  # neutral fallback

        # CB scores: average similarity to user's rated movies
        cb_scores: dict[int, float] = {}
        if user_rated_movies and self.cb._similarity is not None:
            for mid in candidates:
                sims = []
                for rated_mid in user_rated_movies[-50:]:  # cap for speed
                    score_map = self.cb.score_movies([mid], rated_mid)
                    sims.append(score_map.get(mid, 0.0))
                cb_scores[mid] = float(np.mean(sims)) if sims else 0.0
        else:
            cb_scores = {mid: 0.0 for mid in candidates}

        # Blend
        results = []
        for mid in candidates:
            cf_norm = _normalize_cf_score(cf_raw[mid])
            cb = cb_scores[mid]
            hybrid = self.cf_weight * cf_norm + (1 - self.cf_weight) * cb
            results.append(
                {
                    "movie_id": mid,
                    "hybrid_score": round(hybrid, 4),
                    "cf_score": round(cf_raw[mid], 4),
                    "cb_score": round(cb, 4),
                    "strategy": "hybrid",
                }
            )

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_n]

    def _cold_start_recommend(
        self, user_id: int, candidates: list[int], top_n: int
    ) -> list[dict]:
        """For cold-start users: score by CB model only."""
        if self.cb._similarity is None or not candidates:
            return []

        # Pick the most popular candidate as a reference item for similarity
        # (In production this would use user-provided genre preferences)
        results = []
        for mid in candidates:
            similar = self.cb.similar(mid, top_n=1)
            score = similar[0]["similarity_score"] if similar else 0.0
            results.append(
                {
                    "movie_id": mid,
                    "hybrid_score": round(score, 4),
                    "cf_score": None,
                    "cb_score": round(score, 4),
                    "strategy": "cold_start",
                }
            )

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # Similar items (delegates to CB)
    # ------------------------------------------------------------------

    def similar_items(self, movie_id: int, top_n: int = 10) -> list[dict]:
        return self.cb.similar(movie_id, top_n=top_n)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else MODEL_DIR / "hybrid.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "HybridRecommender":
        path = Path(path) if path else MODEL_DIR / "hybrid.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
