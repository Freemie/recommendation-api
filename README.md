# Recommendation System API

A production-grade recommendation engine built with collaborative filtering, content-based filtering, and a hybrid approach — trained on the MovieLens 25M dataset.

## Architecture

```
├── api/          # FastAPI backend (REST endpoints, auth, caching)
├── ml/           # ML models (collaborative filtering, content-based, hybrid)
├── frontend/     # React demo interface
├── docker-compose.yml
└── README.md
```

## Tech Stack

- **API**: FastAPI, PostgreSQL, Redis, JWT auth
- **ML**: Scikit-Surprise (matrix factorization), TF-IDF, MLflow
- **Frontend**: React
- **Infra**: Docker, GitHub Actions CI/CD

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/recommend/{user_id}` | Get personalized recommendations |
| POST | `/feedback` | Record user interactions |
| GET | `/similar/{item_id}` | Find similar items |
| GET | `/trending` | Get trending items |

## Quick Start

```bash
# Clone and start all services
git clone https://github.com/Freemie/recommendation-api
cd recommendation-api
docker-compose up --build

# Or run locally
cd api && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Dataset

Uses [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) — 25 million ratings from 162,000 users on 62,000 movies.

```bash
cd ml && python scripts/ingest.py
```
