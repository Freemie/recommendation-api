from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from core.cache import cache_get, cache_set
from models.user import User
from schemas.recommendation import (
    RecommendResponse, SimilarResponse, TrendingResponse, MoodResponse
)
from services.recommendation_service import (
    get_recommendations, get_similar, get_trending,
    get_discovery, get_mood_recommendations,
    MOOD_GENRE_MAP, ACTIVITY_GENRE_MAP,
)

router = APIRouter()

CACHE_TTL = 300  # 5 minutes


@router.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(
    user_id: int,
    top_n: int = Query(10, ge=1, le=100),
    seed_ids: list[int] = Query(default=[], alias="seed"),
    genres: list[str] = Query(default=[]),
    mood: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get personalised movie recommendations for a user.

    - **seed**: Optional movie IDs to bias recommendations toward similar titles.
    - **genres**: Optional genre filter (e.g. Action, Comedy).
    - **mood**: Optional mood key. See /mood/options for valid values.
    """
    cache_key = f"rec:{user_id}:{top_n}:{sorted(seed_ids)}:{sorted(genres)}:{mood}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    results = get_recommendations(db, user_id, top_n, seed_ids or None, genres or None, mood)
    response = {"user_id": user_id, "count": len(results), "results": results}
    await cache_set(cache_key, response, ttl=CACHE_TTL)
    return response


@router.get("/similar/{movie_id}", response_model=SimilarResponse)
async def similar(
    movie_id: int,
    top_n: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Find movies similar to the given movie (content-based)."""
    cache_key = f"similar:{movie_id}:{top_n}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    results = get_similar(db, movie_id, top_n)
    response = {"movie_id": movie_id, "count": len(results), "results": results}
    await cache_set(cache_key, response, ttl=CACHE_TTL * 6)  # similar items change rarely
    return response


@router.get("/trending", response_model=TrendingResponse)
async def trending(
    top_n: int = Query(20, ge=1, le=100),
    genre: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Get trending movies by rating count, optionally filtered by genre."""
    cache_key = f"trending:{top_n}:{genre}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    results = get_trending(db, top_n, genre)
    response = {"count": len(results), "genre_filter": genre, "results": results}
    await cache_set(cache_key, response, ttl=CACHE_TTL * 12)  # trending changes slowly
    return response


@router.get("/discover/{user_id}", response_model=RecommendResponse)
async def discover(
    user_id: int,
    top_n: int = Query(20, ge=1, le=100),
    discovery_level: float = Query(0.7, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Discovery playlist — mix of familiar picks and new genres.

    - **discovery_level**: 0.0 = all familiar, 1.0 = all new. Default 0.7.
    """
    cache_key = f"discover:{user_id}:{top_n}:{discovery_level}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    results = get_discovery(db, user_id, top_n, discovery_level)
    response = {"user_id": user_id, "count": len(results), "results": results}
    await cache_set(cache_key, response, ttl=CACHE_TTL)
    return response


@router.get("/mood", response_model=MoodResponse)
async def mood_recommendations(
    mood: str | None = Query(default=None),
    activity: str | None = Query(default=None),
    top_n: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Mood or activity-based recommendations.

    Pass either **mood** or **activity**, not both.
    See /mood/options for valid values.
    """
    if not mood and not activity:
        raise ValueError("Provide either 'mood' or 'activity' query parameter.")

    cache_key = f"mood:{mood}:{activity}:{top_n}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    results = get_mood_recommendations(db, mood or "", top_n, activity)
    response = {"mood": mood, "activity": activity, "count": len(results), "results": results}
    await cache_set(cache_key, response, ttl=CACHE_TTL * 6)
    return response


@router.get("/mood/options")
async def mood_options():
    """List all valid mood and activity values."""
    return {
        "moods": list(MOOD_GENRE_MAP.keys()),
        "activities": list(ACTIVITY_GENRE_MAP.keys()),
    }
