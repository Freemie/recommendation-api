from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..core.database import Base


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    tag = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    user = relationship("User", back_populates="tags")
    movie = relationship("Movie", back_populates="tags")

    def __repr__(self):
        return f"<Tag user={self.user_id} movie={self.movie_id} tag={self.tag!r}>"


class GenomeScore(Base):
    """MovieLens genome scores — relevance of each tag to each movie."""
    __tablename__ = "genome_scores"

    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    tag_id = Column(Integer, nullable=False, index=True)    # genome tagId
    tag = Column(String(255), nullable=False)
    relevance = Column(Float, nullable=False)               # 0.0 – 1.0

    def __repr__(self):
        return f"<GenomeScore movie={self.movie_id} tag={self.tag!r} relevance={self.relevance:.3f}>"
