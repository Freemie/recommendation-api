from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from core.cache import cache_delete_pattern
from models.user import User
from models.rating import Rating
from schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    body: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Record a user interaction (rating, click, watch, skip).

    Upserts the rating and invalidates cached recommendations for this user.
    """
    existing = (
        db.query(Rating)
        .filter(Rating.user_id == current_user.id, Rating.movie_id == body.movie_id)
        .first()
    )

    if existing:
        existing.rating = body.rating
        existing.timestamp = datetime.now(timezone.utc)
        db.commit()
        message = "Rating updated"
    else:
        rating = Rating(
            user_id=current_user.id,
            movie_id=body.movie_id,
            rating=body.rating,
            timestamp=datetime.now(timezone.utc),
        )
        db.add(rating)
        db.commit()
        message = "Rating recorded"

    # Invalidate all cached recommendations for this user
    await cache_delete_pattern(f"rec:{current_user.id}:*")
    await cache_delete_pattern(f"discover:{current_user.id}:*")

    return FeedbackResponse(
        user_id=current_user.id,
        movie_id=body.movie_id,
        rating=body.rating,
        message=message,
    )
