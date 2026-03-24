"""
Recommendation service — bridges API layer with ML models.

Incorporates patterns from Music Rec Project:
  - personalized_recommendations: seed-based + genre + mood fallback
  - generate_discovery_playlist: familiar vs. discovery split
  - Mood presets mapped to MovieLens genres
"""

from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import func

from core.model_store import model_store
from models.movie import Movie
from models.rating import Rating


# ---------------------------------------------------------------------------
# Mood → genre mapping (adapted from Music Rec Project Mood_based.py)
# MovieLens doesn't have audio features, so we map mood to genres.
# ---------------------------------------------------------------------------

MOOD_GENRE_MAP: dict[str, list[str]] = {
    "happy":      ["Comedy", "Animation", "Family", "Musical"],
    "sad":        ["Drama", "Romance"],
    "energetic":  ["Action", "Adventure", "Thriller"],
    "relaxed":    ["Documentary", "Music", "Western"],
    "focus":      ["Documentary", "Biography", "History"],
    "romantic":   ["Romance", "Drama"],
    "scary":      ["Horror", "Thriller", "Mystery"],
    "nostalgic":  ["Animation", "Family", "Musical", "Classic"],
    "discovery":  ["Foreign", "Art", "Independent"],
    "party":      ["Comedy", "Musical", "Animation"],
}

ACTIVITY_GENRE_MAP: dict[str, list[str]] = {
    "commuting":  ["Comedy", "Action"],
    "studying":   ["Documentary", "Biography"],
    "working_out": ["Action", "Adventure", "Thriller"],
    "date_night": ["Romance", "Comedy", "Drama"],
    "family":     ["Animation", "Family", "Adventure"],
    "winding_down": ["Drama", "Documentary"],
}


def _get_user_rated_movie_ids(db: Session, user_id: int) -> list[int]:
    rows = db.query(Rating.movie_id).filter(Rating.user_id == user_id).all()
    return [r.movie_id for r in rows]


def _get_candidate_pool(db: Session, limit: int = 5000) -> list[int]:
    """Return the top-rated movies by rating count as the default candidate pool."""
    rows = (
        db.query(Movie.id)
        .filter(Movie.rating_count > 0)
        .order_by(Movie.rating_count.desc())
        .limit(limit)
        .all()
    )
    return [r.id for r in rows]


def _enrich_with_metadata(db: Session, results: list[dict]) -> list[dict]:
    """Attach title and genres to each recommendation dict."""
    movie_ids = [r["movie_id"] for r in results]
    movies = db.query(Movie).filter(Movie.id.in_(movie_ids)).all()
    meta = {m.id: m for m in movies}
    for r in results:
        m = meta.get(r["movie_id"])
        if m:
            r["title"] = m.title
            r["genres"] = m.genre_list()
            r["avg_rating"] = m.avg_rating
    return results


# ---------------------------------------------------------------------------
# Core recommendation functions
# ---------------------------------------------------------------------------

def get_recommendations(
    db: Session,
    user_id: int,
    top_n: int = 10,
    seed_movie_ids: list[int] | None = None,
    genres: list[str] | None = None,
    mood: str | None = None,
) -> list[dict]:
    """
    Personalised recommendations for a user.

    Incorporates Music Rec Project's personalized_recommendations strategy:
      1. If models are ready → hybrid CF + CB
      2. Seed movies provided → boost CB similarity to seeds
      3. Mood/genre filter → narrow candidate pool first
      4. Cold-start fallback → popularity + genre match

    Args:
        db: DB session.
        user_id: Target user.
        top_n: Number of results.
        seed_movie_ids: Optional seed movies to bias recommendations.
        genres: Optional genre filter.
        mood: Optional mood key (see MOOD_GENRE_MAP).
    """
    already_rated = set(_get_user_rated_movie_ids(db, user_id))

    # Resolve genre filter from mood or explicit genres
    target_genres: list[str] = []
    if mood and mood.lower() in MOOD_GENRE_MAP:
        target_genres = MOOD_GENRE_MAP[mood.lower()]
    elif genres:
        target_genres = genres

    # Build candidate pool — filter by genre if requested
    if target_genres:
        genre_filter = [f"%{g}%" for g in target_genres]
        from sqlalchemy import or_
        candidates_q = (
            db.query(Movie.id)
            .filter(
                or_(*[Movie.genres.ilike(p) for p in genre_filter]),
                Movie.rating_count > 10,
            )
            .order_by(Movie.rating_count.desc())
            .limit(3000)
        )
        candidate_ids = [r.id for r in candidates_q.all()]
    else:
        candidate_ids = _get_candidate_pool(db, limit=5000)

    if not model_store.is_ready:
        return _popularity_fallback(db, candidate_ids, already_rated, top_n)

    hybrid = model_store.hybrid

    # Seed-based boost: if seeds provided, re-rank candidates by CB similarity
    # (from Music Rec Project personalized_recommendations seed logic)
    if seed_movie_ids:
        candidate_ids = _rerank_by_seeds(seed_movie_ids, candidate_ids, top_n * 5)

    results = hybrid.recommend(
        user_id=user_id,
        candidate_movie_ids=candidate_ids,
        top_n=top_n,
        already_rated=already_rated,
        user_rated_movies=list(already_rated),
    )

    return _enrich_with_metadata(db, results)


def _rerank_by_seeds(
    seed_ids: list[int], candidates: list[int], limit: int
) -> list[int]:
    """
    Re-rank candidates by average CB similarity to seed movies.
    Mirrors Music Rec Project's seed-based get_content_recommendations loop.
    """
    if not model_store.cb or not seed_ids:
        return candidates

    scores: dict[int, float] = {}
    for seed_id in seed_ids[:3]:  # cap at 3 seeds (same as Music Rec Project)
        sim_map = model_store.cb.score_movies(candidates, seed_id)
        for mid, score in sim_map.items():
            scores[mid] = scores.get(mid, 0.0) + score / len(seed_ids)

    reranked = sorted(candidates, key=lambda m: scores.get(m, 0.0), reverse=True)
    return reranked[:limit]


def get_similar(db: Session, movie_id: int, top_n: int = 10) -> list[dict]:
    """Find movies similar to a given movie via content-based filtering."""
    if not model_store.is_ready:
        raise ValueError("Models not loaded yet.")

    results = model_store.cb.similar(movie_id, top_n=top_n)
    return _enrich_with_metadata(db, results)


def get_trending(db: Session, top_n: int = 20, genre: str | None = None) -> list[dict]:
    """Return top movies by rating count, optionally filtered by genre."""
    q = db.query(Movie).filter(Movie.rating_count > 0)
    if genre:
        q = q.filter(Movie.genres.ilike(f"%{genre}%"))
    movies = q.order_by(Movie.rating_count.desc()).limit(top_n).all()
    return [
        {
            "movie_id": m.id,
            "title": m.title,
            "genres": m.genre_list(),
            "avg_rating": m.avg_rating,
            "rating_count": m.rating_count,
        }
        for m in movies
    ]


def get_discovery(
    db: Session,
    user_id: int,
    top_n: int = 20,
    discovery_level: float = 0.7,
) -> list[dict]:
    """
    Discovery playlist: mix of familiar and new recommendations.

    Directly adapted from Music Rec Project's generate_discovery_playlist —
    splits top_n into familiar (similar to history) and discovery (different).

    Args:
        discovery_level: 0.0 = all familiar, 1.0 = all new. Default 0.7.
    """
    familiar_count = max(1, int(top_n * (1 - discovery_level)))
    discovery_count = top_n - familiar_count

    rated_ids = _get_user_rated_movie_ids(db, user_id)
    already_rated = set(rated_ids)

    results = []

    # Familiar: recommendations similar to recently rated movies
    if familiar_count > 0 and rated_ids and model_store.is_ready:
        for seed_id in rated_ids[-3:]:  # last 3 rated (same as Music Rec Project)
            similar = model_store.cb.similar(seed_id, top_n=familiar_count, exclude_ids=already_rated)
            for s in similar:
                s["recommendation_source"] = "familiar"
            results.extend(similar)

    # Discovery: different genres from user history
    if discovery_count > 0:
        # Find genres the user has NOT rated much
        rated_genres = _get_user_top_genres(db, user_id)
        from sqlalchemy import and_, not_, or_
        discovery_q = (
            db.query(Movie.id)
            .filter(
                Movie.id.notin_(already_rated),
                Movie.rating_count > 50,
                not_(or_(*[Movie.genres.ilike(f"%{g}%") for g in rated_genres[:3]])) if rated_genres else True,
            )
            .order_by(Movie.rating_count.desc())
            .limit(discovery_count * 3)
        )
        discovery_ids = [r.id for r in discovery_q.all()]
        disc_movies = db.query(Movie).filter(Movie.id.in_(discovery_ids[:discovery_count])).all()
        for m in disc_movies:
            results.append({
                "movie_id": m.id,
                "title": m.title,
                "genres": m.genre_list(),
                "avg_rating": m.avg_rating,
                "recommendation_source": "discovery",
            })

    # Deduplicate and shuffle (mirrors Music Rec Project)
    seen: set[int] = set()
    unique = []
    for r in results:
        mid = r.get("movie_id")
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(r)

    import random
    random.shuffle(unique)
    return unique[:top_n]


def get_mood_recommendations(
    db: Session,
    mood: str,
    top_n: int = 20,
    activity: str | None = None,
) -> list[dict]:
    """
    Mood/activity-based recommendations using genre mapping.
    Adapted from Music Rec Project's create_mood_based_playlist and
    create_activity_playlist.
    """
    genre_map = ACTIVITY_GENRE_MAP if activity else MOOD_GENRE_MAP
    key = (activity or mood).lower()

    if key not in genre_map:
        valid = list(genre_map.keys())
        raise ValueError(f"'{key}' not recognised. Valid options: {valid}")

    target_genres = genre_map[key]
    from sqlalchemy import or_
    movies = (
        db.query(Movie)
        .filter(
            or_(*[Movie.genres.ilike(f"%{g}%") for g in target_genres]),
            Movie.rating_count > 50,
        )
        .order_by(Movie.avg_rating.desc(), Movie.rating_count.desc())
        .limit(top_n)
        .all()
    )
    return [
        {
            "movie_id": m.id,
            "title": m.title,
            "genres": m.genre_list(),
            "avg_rating": m.avg_rating,
            "mood": mood,
            "activity": activity,
            "recommendation_source": f"mood:{key}",
        }
        for m in movies
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _popularity_fallback(
    db: Session,
    candidate_ids: list[int],
    exclude: set[int],
    top_n: int,
) -> list[dict]:
    """Used when models aren't loaded yet."""
    ids = [mid for mid in candidate_ids if mid not in exclude][:top_n * 2]
    movies = db.query(Movie).filter(Movie.id.in_(ids)).order_by(Movie.rating_count.desc()).limit(top_n).all()
    return [
        {
            "movie_id": m.id,
            "title": m.title,
            "genres": m.genre_list(),
            "avg_rating": m.avg_rating,
            "strategy": "popularity_fallback",
        }
        for m in movies
    ]


def _get_user_top_genres(db: Session, user_id: int, top_n: int = 5) -> list[str]:
    """Return genres the user has rated most (for discovery filtering)."""
    rated = (
        db.query(Movie.genres)
        .join(Rating, Movie.id == Rating.movie_id)
        .filter(Rating.user_id == user_id, Rating.rating >= 3.5)
        .all()
    )
    genre_count: dict[str, int] = {}
    for (genres_str,) in rated:
        for g in genres_str.split("|"):
            genre_count[g] = genre_count.get(g, 0) + 1
    return sorted(genre_count, key=genre_count.get, reverse=True)[:top_n]
