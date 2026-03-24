"""
MovieLens 25M data ingestion pipeline.

Downloads the dataset, preprocesses it, and loads it into PostgreSQL.

Usage:
    python -m ml.scripts.ingest [--limit 100000]
"""

import argparse
import io
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))

from sqlalchemy.orm import Session
from core.database import engine, Base
from models import User, Movie, Rating, Tag, GenomeScore

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ZIP_PATH = DATA_DIR / "ml-25m.zip"
EXTRACT_DIR = DATA_DIR / "ml-25m"

BATCH_SIZE = 10_000


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists():
        print("Dataset already extracted, skipping download.")
        return

    if not ZIP_PATH.exists():
        print(f"Downloading MovieLens 25M (~250 MB) from {MOVIELENS_URL} ...")
        response = requests.get(MOVIELENS_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(ZIP_PATH, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
        print("Download complete.")

    print("Extracting archive ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(unix_ts: str) -> datetime:
    return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)


def _bulk_insert(session: Session, objects: list, label: str):
    session.bulk_save_objects(objects)
    session.commit()
    print(f"  Inserted {len(objects):,} {label}.")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_movies(session: Session):
    import csv
    path = EXTRACT_DIR / "movies.csv"
    print(f"Loading movies from {path} ...")

    # Load links for imdb/tmdb IDs
    links: dict[int, tuple[str, int]] = {}
    links_path = EXTRACT_DIR / "links.csv"
    if links_path.exists():
        with open(links_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = int(row["movieId"])
                imdb_id = row.get("imdbId", "")
                tmdb_id = int(row["tmdbId"]) if row.get("tmdbId") else None
                links[movie_id] = (imdb_id, tmdb_id)

    batch = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = int(row["movieId"])
            imdb_id, tmdb_id = links.get(movie_id, ("", None))
            batch.append(Movie(
                id=movie_id,
                title=row["title"],
                genres=row["genres"],
                imdb_id=imdb_id or None,
                tmdb_id=tmdb_id,
            ))
            if len(batch) >= BATCH_SIZE:
                _bulk_insert(session, batch, "movies")
                batch = []
    if batch:
        _bulk_insert(session, batch, "movies")


def load_users(session: Session, limit: int | None = None):
    """Derive unique users from ratings.csv and insert them."""
    import csv
    path = EXTRACT_DIR / "ratings.csv"
    print("Deriving unique users from ratings ...")

    seen: set[int] = set()
    batch: list[User] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            uid = int(row["userId"])
            if uid not in seen:
                seen.add(uid)
                batch.append(User(movielens_id=uid))
                if len(batch) >= BATCH_SIZE:
                    _bulk_insert(session, batch, "users")
                    batch = []
    if batch:
        _bulk_insert(session, batch, "users")

    # Build movielens_id -> db id map
    print("Building user id map ...")
    rows = session.query(User.id, User.movielens_id).all()
    return {ml_id: db_id for db_id, ml_id in rows}


def load_ratings(session: Session, user_map: dict[int, int], limit: int | None = None):
    import csv
    path = EXTRACT_DIR / "ratings.csv"
    print(f"Loading ratings from {path} ...")

    batch = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            db_user_id = user_map.get(int(row["userId"]))
            if db_user_id is None:
                continue
            batch.append(Rating(
                user_id=db_user_id,
                movie_id=int(row["movieId"]),
                rating=float(row["rating"]),
                timestamp=_ts(row["timestamp"]),
            ))
            if len(batch) >= BATCH_SIZE:
                _bulk_insert(session, batch, "ratings")
                batch = []
    if batch:
        _bulk_insert(session, batch, "ratings")


def load_tags(session: Session, user_map: dict[int, int], limit: int | None = None):
    import csv
    path = EXTRACT_DIR / "tags.csv"
    print(f"Loading tags from {path} ...")

    batch = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            db_user_id = user_map.get(int(row["userId"]))
            if db_user_id is None:
                continue
            batch.append(Tag(
                user_id=db_user_id,
                movie_id=int(row["movieId"]),
                tag=row["tag"],
                timestamp=_ts(row["timestamp"]),
            ))
            if len(batch) >= BATCH_SIZE:
                _bulk_insert(session, batch, "tags")
                batch = []
    if batch:
        _bulk_insert(session, batch, "tags")


def load_genome_scores(session: Session):
    import csv
    scores_path = EXTRACT_DIR / "genome-scores.csv"
    tags_path = EXTRACT_DIR / "genome-tags.csv"

    if not scores_path.exists():
        print("Genome scores not found, skipping.")
        return

    print("Loading genome tag labels ...")
    tag_labels: dict[int, str] = {}
    with open(tags_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag_labels[int(row["tagId"])] = row["tag"]

    print(f"Loading genome scores from {scores_path} ...")
    batch = []
    with open(scores_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row["tagId"])
            batch.append(GenomeScore(
                movie_id=int(row["movieId"]),
                tag_id=tag_id,
                tag=tag_labels.get(tag_id, ""),
                relevance=float(row["relevance"]),
            ))
            if len(batch) >= BATCH_SIZE:
                _bulk_insert(session, batch, "genome scores")
                batch = []
    if batch:
        _bulk_insert(session, batch, "genome scores")


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

def update_movie_stats(session: Session):
    """Compute and store avg_rating and rating_count for each movie."""
    print("Updating movie rating stats ...")
    from sqlalchemy import func
    results = (
        session.query(
            Rating.movie_id,
            func.avg(Rating.rating).label("avg"),
            func.count(Rating.id).label("cnt"),
        )
        .group_by(Rating.movie_id)
        .all()
    )
    for movie_id, avg, cnt in tqdm(results, desc="  stats"):
        session.query(Movie).filter(Movie.id == movie_id).update(
            {"avg_rating": round(float(avg), 4), "rating_count": cnt}
        )
    session.commit()
    print(f"  Updated stats for {len(results):,} movies.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest MovieLens 25M into PostgreSQL")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows loaded for ratings/tags (useful for dev). Default: load all.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (dataset already present).",
    )
    parser.add_argument(
        "--skip-genome",
        action="store_true",
        help="Skip genome scores (large table, optional for dev).",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset()

    print("\nCreating database tables ...")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        load_movies(session)
        user_map = load_users(session, limit=args.limit)
        load_ratings(session, user_map, limit=args.limit)
        load_tags(session, user_map, limit=args.limit)
        if not args.skip_genome:
            load_genome_scores(session)
        update_movie_stats(session)

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
