"""
Collaborative Filtering via SVD matrix factorization (scikit-surprise).

Trains on (user_id, movie_id, rating) triplets and predicts ratings
for unseen user-movie pairs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class SVDConfig(NamedTuple):
    n_factors: int = 100
    n_epochs: int = 20
    lr_all: float = 0.005
    reg_all: float = 0.02
    random_state: int = 42


class CollaborativeFilter:
    """SVD-based collaborative filtering recommender."""

    def __init__(self, config: SVDConfig | None = None):
        self.config = config or SVDConfig()
        self.model: SVD | None = None
        self.trainset = None
        self._rating_scale = (0.5, 5.0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, ratings_df: pd.DataFrame) -> "CollaborativeFilter":
        """
        Train on a DataFrame with columns: user_id, movie_id, rating.
        """
        reader = Reader(rating_scale=self._rating_scale)
        data = Dataset.load_from_df(
            ratings_df[["user_id", "movie_id", "rating"]], reader
        )
        self.trainset = data.build_full_trainset()

        self.model = SVD(
            n_factors=self.config.n_factors,
            n_epochs=self.config.n_epochs,
            lr_all=self.config.lr_all,
            reg_all=self.config.reg_all,
            random_state=self.config.random_state,
            verbose=True,
        )
        self.model.fit(self.trainset)
        return self

    def fit_with_split(
        self, ratings_df: pd.DataFrame, test_size: float = 0.2
    ) -> tuple["CollaborativeFilter", pd.DataFrame]:
        """
        Train with a held-out test set. Returns (fitted model, test_df).
        test_df has columns: user_id, movie_id, rating, predicted_rating.
        """
        reader = Reader(rating_scale=self._rating_scale)
        data = Dataset.load_from_df(
            ratings_df[["user_id", "movie_id", "rating"]], reader
        )
        trainset, testset = train_test_split(
            data, test_size=test_size, random_state=self.config.random_state
        )
        self.trainset = trainset

        self.model = SVD(
            n_factors=self.config.n_factors,
            n_epochs=self.config.n_epochs,
            lr_all=self.config.lr_all,
            reg_all=self.config.reg_all,
            random_state=self.config.random_state,
            verbose=True,
        )
        self.model.fit(trainset)

        predictions = self.model.test(testset)
        test_df = pd.DataFrame(
            [
                {
                    "user_id": p.uid,
                    "movie_id": p.iid,
                    "rating": p.r_ui,
                    "predicted_rating": p.est,
                }
                for p in predictions
            ]
        )
        return self, test_df

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a single user-movie pair."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(user_id, movie_id).est

    def recommend(
        self,
        user_id: int,
        candidate_movie_ids: list[int],
        top_n: int = 10,
        already_rated: set[int] | None = None,
    ) -> list[dict]:
        """
        Return top-N movie recommendations for a user.

        Args:
            user_id: Target user.
            candidate_movie_ids: Pool of movies to score.
            top_n: Number of results to return.
            already_rated: Movie IDs the user has already rated (excluded).

        Returns:
            List of dicts: {movie_id, predicted_rating}, sorted desc.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        exclude = already_rated or set()
        scores = [
            {"movie_id": mid, "predicted_rating": self.model.predict(user_id, mid).est}
            for mid in candidate_movie_ids
            if mid not in exclude
        ]
        scores.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return scores[:top_n]

    def get_user_embedding(self, user_id: int) -> np.ndarray | None:
        """Return the latent factor vector for a user (for hybrid blending)."""
        if self.model is None or self.trainset is None:
            return None
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            return self.model.pu[inner_uid]
        except ValueError:
            return None  # unknown user

    def get_item_embedding(self, movie_id: int) -> np.ndarray | None:
        """Return the latent factor vector for an item (for hybrid blending)."""
        if self.model is None or self.trainset is None:
            return None
        try:
            inner_iid = self.trainset.to_inner_iid(movie_id)
            return self.model.qi[inner_iid]
        except ValueError:
            return None  # unknown item

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else MODEL_DIR / "collaborative_svd.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "CollaborativeFilter":
        path = Path(path) if path else MODEL_DIR / "collaborative_svd.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
