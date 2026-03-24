from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    movie_id: int
    title: str | None = None
    genres: list[str] = []
    avg_rating: float | None = None
    hybrid_score: float | None = None
    cf_score: float | None = None
    cb_score: float | None = None
    similarity_score: float | None = None
    strategy: str | None = None
    recommendation_source: str | None = None


class RecommendResponse(BaseModel):
    user_id: int | None = None
    count: int
    results: list[RecommendationItem]


class SimilarResponse(BaseModel):
    movie_id: int
    count: int
    results: list[RecommendationItem]


class TrendingItem(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    avg_rating: float | None = None
    rating_count: int | None = None


class TrendingResponse(BaseModel):
    count: int
    genre_filter: str | None = None
    results: list[TrendingItem]


class MoodResponse(BaseModel):
    mood: str | None = None
    activity: str | None = None
    count: int
    results: list[RecommendationItem]
