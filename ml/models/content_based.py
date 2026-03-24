"""
Content-Based Filtering using TF-IDF on movie metadata.

Builds a movie-to-movie similarity matrix from:
  - Title tokens
  - Genre labels
  - MovieLens genome tag scores (when available)

Supports /similar/{movie_id} and cold-start recommendations.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _clean_title(title: str) -> str:
    """Strip year suffix like '(1995)' from title."""
    import re
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def _genres_to_text(genres: str) -> str:
    """'Action|Comedy|Drama' -> 'action comedy drama'"""
    return genres.replace("|", " ").replace("-", "").lower()


class ContentBasedFilter:
    """
    Item-to-item recommender based on TF-IDF movie metadata.

    After calling fit(), use:
      - similar(movie_id, top_n) — find similar movies
      - recommend_for_genres(genres, top_n) — cold-start by genre
    """

    def __init__(
        self,
        use_genome: bool = True,
        genre_weight: float = 2.0,
        genome_weight: float = 1.5,
    ):
        self.use_genome = use_genome
        self.genre_weight = genre_weight
        self.genome_weight = genome_weight

        self._movie_ids: list[int] = []
        self._id_to_idx: dict[int, int] = {}
        self._similarity: np.ndarray | None = None
        self._movies_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        movies_df: pd.DataFrame,
        genome_df: pd.DataFrame | None = None,
    ) -> "ContentBasedFilter":
        """
        Build the similarity matrix.

        Args:
            movies_df: DataFrame with columns [id, title, genres].
            genome_df: Optional DataFrame with columns [movie_id, tag, relevance].
                       If provided, top genome tags are appended to the text.
        """
        movies_df = movies_df.copy().reset_index(drop=True)
        self._movies_df = movies_df
        self._movie_ids = movies_df["id"].tolist()
        self._id_to_idx = {mid: i for i, mid in enumerate(self._movie_ids)}

        # Build text corpus: title + genres (weighted by repetition)
        corpus = []
        for _, row in movies_df.iterrows():
            title_text = _clean_title(str(row["title"]))
            genre_text = " ".join(
                [_genres_to_text(str(row["genres"]))] * int(self.genre_weight)
            )
            corpus.append(f"{title_text} {genre_text}")

        # TF-IDF on text features
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20_000,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        self._vectorizer = vectorizer

        feature_matrix = normalize(tfidf_matrix)

        # Append genome tag features if available
        if self.use_genome and genome_df is not None and not genome_df.empty:
            genome_matrix = self._build_genome_matrix(genome_df, movies_df)
            if genome_matrix is not None:
                genome_matrix = normalize(genome_matrix) * self.genome_weight
                from scipy.sparse import hstack, issparse
                if issparse(feature_matrix):
                    feature_matrix = hstack([feature_matrix, genome_matrix])
                else:
                    feature_matrix = np.hstack([feature_matrix, genome_matrix.toarray()])

        # Use batched similarity for memory efficiency on large catalogs
        # (adapted from Music Rec Project Similarity_calculator.batch_calculate_similarities)
        self._similarity = self._batch_cosine_similarity(feature_matrix)
        return self

    def _batch_cosine_similarity(
        self, feature_matrix, batch_size: int = 1000
    ) -> "scipy.sparse.csr_matrix":
        """
        Compute cosine similarity in row batches to avoid OOM on large matrices.
        Returns a sparse matrix (same interface as cosine_similarity dense=False).
        Adapted from Music Rec Project's batch_calculate_similarities.
        """
        from scipy.sparse import csr_matrix, vstack as sparse_vstack, issparse
        import time

        n = feature_matrix.shape[0]
        start = time.time()
        rows = []

        for i in range(0, n, batch_size):
            batch = feature_matrix[i: i + batch_size]
            if issparse(batch):
                batch_sim = cosine_similarity(batch, feature_matrix, dense_output=False)
            else:
                batch_sim = cosine_similarity(batch, feature_matrix, dense_output=False)
            rows.append(batch_sim)

        result = sparse_vstack(rows)
        print(f"Similarity matrix built in {time.time() - start:.1f}s — shape {result.shape}")
        return result

    def _build_genome_matrix(
        self, genome_df: pd.DataFrame, movies_df: pd.DataFrame
    ) -> "scipy.sparse.csr_matrix | None":
        """Pivot genome scores into a (n_movies, n_tags) sparse matrix."""
        try:
            from scipy.sparse import csr_matrix

            merged = genome_df[genome_df["movie_id"].isin(self._movie_ids)]
            if merged.empty:
                return None

            pivot = merged.pivot_table(
                index="movie_id", columns="tag_id", values="relevance", fill_value=0.0
            )
            # Align rows to movies_df order
            pivot = pivot.reindex(self._movie_ids, fill_value=0.0)
            return csr_matrix(pivot.values)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def similar(
        self,
        movie_id: int,
        top_n: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[dict]:
        """
        Return top-N movies most similar to movie_id.

        Returns:
            List of dicts: {movie_id, title, genres, similarity_score}
        """
        if self._similarity is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        idx = self._id_to_idx.get(movie_id)
        if idx is None:
            return []

        exclude = (exclude_ids or set()) | {movie_id}
        sim_row = np.asarray(self._similarity[idx].todense()).flatten()

        # Get top candidates (overshoot to account for exclusions)
        candidate_indices = np.argsort(sim_row)[::-1][: top_n * 3]

        results = []
        for i in candidate_indices:
            mid = self._movie_ids[i]
            if mid in exclude:
                continue
            row = self._movies_df.iloc[i]
            results.append(
                {
                    "movie_id": mid,
                    "title": row["title"],
                    "genres": row["genres"],
                    "similarity_score": float(sim_row[i]),
                }
            )
            if len(results) >= top_n:
                break

        return results

    def score_movies(self, movie_ids: list[int], target_movie_id: int) -> dict[int, float]:
        """
        Return similarity scores for a list of movies against a target movie.
        Used by the hybrid recommender.
        """
        if self._similarity is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        target_idx = self._id_to_idx.get(target_movie_id)
        if target_idx is None:
            return {}

        sim_row = np.asarray(self._similarity[target_idx].todense()).flatten()
        return {
            mid: float(sim_row[self._id_to_idx[mid]])
            for mid in movie_ids
            if mid in self._id_to_idx
        }

    def recommend_for_genres(self, genres: list[str], top_n: int = 10) -> list[dict]:
        """
        Cold-start: recommend movies matching a genre list.
        Used when a user has no rating history.
        """
        if self._movies_df is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        genre_set = {g.lower() for g in genres}
        matches = self._movies_df[
            self._movies_df["genres"].apply(
                lambda g: bool(genre_set & {x.lower() for x in g.split("|")})
            )
        ]
        return [
            {
                "movie_id": int(row["id"]),
                "title": row["title"],
                "genres": row["genres"],
                "similarity_score": 1.0,
            }
            for _, row in matches.head(top_n).iterrows()
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else MODEL_DIR / "content_based.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ContentBasedFilter":
        path = Path(path) if path else MODEL_DIR / "content_based.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
