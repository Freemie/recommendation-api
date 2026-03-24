from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..core.database import Base


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)          # MovieLens movieId
    title = Column(String(500), nullable=False, index=True)
    genres = Column(String(500), nullable=False)                 # Pipe-separated: "Action|Comedy"
    imdb_id = Column(String(20), nullable=True)
    tmdb_id = Column(Integer, nullable=True, index=True)
    avg_rating = Column(Float, nullable=True)
    rating_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    ratings = relationship("Rating", back_populates="movie", lazy="dynamic")
    tags = relationship("Tag", back_populates="movie", lazy="dynamic")

    def genre_list(self) -> list[str]:
        return self.genres.split("|") if self.genres else []

    def __repr__(self):
        return f"<Movie id={self.id} title={self.title!r}>"
