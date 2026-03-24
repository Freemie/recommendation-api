"""
Training pipeline with MLflow experiment tracking.

Trains collaborative filtering, content-based, and hybrid models,
logs all params/metrics/artifacts to MLflow, and saves model artifacts.

Usage:
    python -m ml.scripts.train [options]

    --cf-only       Train only collaborative filtering
    --cb-only       Train only content-based
    --limit N       Use N ratings (dev mode)
    --cf-factors N  SVD latent factors (default 100)
    --cf-epochs N   SVD training epochs (default 20)
    --cf-weight F   Hybrid CF weight 0-1 (default 0.7)
    --eval-k N      Ranking cutoff for eval metrics (default 10)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd

# Allow running as a standalone script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.models.collaborative import CollaborativeFilter, SVDConfig
from ml.models.content_based import ContentBasedFilter
from ml.models.hybrid import HybridRecommender
from ml.models.evaluate import evaluate_predictions, evaluate_ranking

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-25m"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ratings(limit: int | None = None) -> pd.DataFrame:
    path = DATA_DIR / "ratings.csv"
    print(f"Loading ratings from {path} ...")
    df = pd.read_csv(path, nrows=limit)
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    print(f"  Loaded {len(df):,} ratings.")
    return df


def load_movies() -> pd.DataFrame:
    path = DATA_DIR / "movies.csv"
    print(f"Loading movies from {path} ...")
    df = pd.read_csv(path)
    df = df.rename(columns={"movieId": "id"})
    print(f"  Loaded {len(df):,} movies.")
    return df


def load_genome(limit_movies: int | None = None) -> pd.DataFrame | None:
    scores_path = DATA_DIR / "genome-scores.csv"
    tags_path = DATA_DIR / "genome-tags.csv"
    if not scores_path.exists():
        print("Genome scores not found, skipping.")
        return None

    print("Loading genome scores ...")
    tags_df = pd.read_csv(tags_path).rename(columns={"tagId": "tag_id"})
    scores_df = pd.read_csv(scores_path).rename(
        columns={"movieId": "movie_id", "tagId": "tag_id"}
    )
    # Keep only high-relevance tags to reduce size
    scores_df = scores_df[scores_df["relevance"] >= 0.5]
    scores_df = scores_df.merge(tags_df[["tag_id", "tag"]], on="tag_id", how="left")
    print(f"  Loaded {len(scores_df):,} genome score rows.")
    return scores_df


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def train_collaborative(
    ratings_df: pd.DataFrame,
    config: SVDConfig,
    run: mlflow.ActiveRun,
) -> tuple[CollaborativeFilter, pd.DataFrame]:
    print("\n--- Training Collaborative Filter (SVD) ---")
    mlflow.log_params({
        "cf_n_factors": config.n_factors,
        "cf_n_epochs": config.n_epochs,
        "cf_lr_all": config.lr_all,
        "cf_reg_all": config.reg_all,
    })

    cf = CollaborativeFilter(config)
    cf, test_df = cf.fit_with_split(ratings_df, test_size=0.2)

    metrics = evaluate_predictions(test_df)
    print(f"  RMSE: {metrics['rmse']}  MAE: {metrics['mae']}")
    mlflow.log_metrics({"cf_rmse": metrics["rmse"], "cf_mae": metrics["mae"]})

    model_path = cf.save()
    mlflow.log_artifact(str(model_path), artifact_path="models")
    print(f"  Saved: {model_path}")

    return cf, test_df


def train_content_based(
    movies_df: pd.DataFrame,
    genome_df: pd.DataFrame | None,
    run: mlflow.ActiveRun,
) -> ContentBasedFilter:
    print("\n--- Training Content-Based Filter ---")
    mlflow.log_params({
        "cb_use_genome": genome_df is not None,
        "cb_genre_weight": 2.0,
        "cb_genome_weight": 1.5,
    })

    cb = ContentBasedFilter(use_genome=genome_df is not None)
    cb.fit(movies_df, genome_df)

    model_path = cb.save()
    mlflow.log_artifact(str(model_path), artifact_path="models")
    print(f"  Saved: {model_path}")

    return cb


def train_hybrid(
    cf: CollaborativeFilter,
    cb: ContentBasedFilter,
    cf_weight: float,
    test_df: pd.DataFrame,
    all_movie_ids: list[int],
    eval_k: int,
    run: mlflow.ActiveRun,
) -> HybridRecommender:
    print("\n--- Building Hybrid Recommender ---")
    mlflow.log_param("hybrid_cf_weight", cf_weight)

    hybrid = HybridRecommender(cf, cb, cf_weight=cf_weight)

    print(f"  Evaluating ranking metrics (k={eval_k}, max 200 users) ...")
    ranking_metrics = evaluate_ranking(
        hybrid,
        test_df,
        all_movie_ids,
        k=eval_k,
        n_users=200,
    )
    print(f"  {ranking_metrics}")
    mlflow.log_metrics({
        f"hybrid_precision_at_{eval_k}": ranking_metrics[f"precision@{eval_k}"],
        f"hybrid_recall_at_{eval_k}": ranking_metrics[f"recall@{eval_k}"],
        f"hybrid_ndcg_at_{eval_k}": ranking_metrics[f"ndcg@{eval_k}"],
    })

    model_path = hybrid.save()
    mlflow.log_artifact(str(model_path), artifact_path="models")
    print(f"  Saved: {model_path}")

    return hybrid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument("--cf-only", action="store_true")
    parser.add_argument("--cb-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of ratings loaded (dev mode)")
    parser.add_argument("--cf-factors", type=int, default=100)
    parser.add_argument("--cf-epochs", type=int, default=20)
    parser.add_argument("--cf-weight", type=float, default=0.7)
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("recommendation-system")

    run_name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"\nStarting MLflow run: {run_name}")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}\n")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("limit", args.limit)
        mlflow.log_param("eval_k", args.eval_k)

        movies_df = load_movies()
        all_movie_ids = movies_df["id"].tolist()

        cf_model, test_df = None, None
        cb_model = None

        if not args.cb_only:
            ratings_df = load_ratings(limit=args.limit)
            config = SVDConfig(
                n_factors=args.cf_factors,
                n_epochs=args.cf_epochs,
            )
            cf_model, test_df = train_collaborative(ratings_df, config, run)

        if not args.cf_only:
            genome_df = load_genome()
            cb_model = train_content_based(movies_df, genome_df, run)

        if cf_model and cb_model and test_df is not None:
            train_hybrid(
                cf_model, cb_model,
                cf_weight=args.cf_weight,
                test_df=test_df,
                all_movie_ids=all_movie_ids,
                eval_k=args.eval_k,
                run=run,
            )

        print(f"\nRun ID: {run.info.run_id}")
        print("Training complete.")


if __name__ == "__main__":
    main()
