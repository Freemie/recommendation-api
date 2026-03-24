from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    # MovieLens userId — for imported users; null for registered users
    movielens_id = Column(Integer, nullable=True, unique=True, index=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    hashed_password = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    ratings = relationship("Rating", back_populates="user", lazy="dynamic")
    tags = relationship("Tag", back_populates="user", lazy="dynamic")

    def __repr__(self):
        return f"<User id={self.id} email={self.email!r}>"
