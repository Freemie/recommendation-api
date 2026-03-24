from pydantic import BaseModel, Field
from typing import Literal


class FeedbackRequest(BaseModel):
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating from 0.5 to 5.0")
    interaction_type: Literal["rating", "click", "watch", "skip"] = "rating"


class FeedbackResponse(BaseModel):
    user_id: int
    movie_id: int
    rating: float
    message: str
