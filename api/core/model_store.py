"""
Singleton that holds loaded ML models for the lifetime of the API process.
Models are loaded once at startup (lifespan) and reused across requests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing ml package from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "ml" / "models" / "artifacts"


class ModelStore:
    def __init__(self):
        self.hybrid = None
        self.cf = None
        self.cb = None
        self.is_ready = False

    async def load(self):
        """Load all trained models from disk (non-blocking wrapper)."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        try:
            from ml.models.hybrid import HybridRecommender
            from ml.models.collaborative import CollaborativeFilter
            from ml.models.content_based import ContentBasedFilter

            hybrid_path = ARTIFACT_DIR / "hybrid.pkl"
            cf_path = ARTIFACT_DIR / "collaborative_svd.pkl"
            cb_path = ARTIFACT_DIR / "content_based.pkl"

            if hybrid_path.exists():
                self.hybrid = HybridRecommender.load(hybrid_path)
                self.cf = self.hybrid.cf
                self.cb = self.hybrid.cb
                self.is_ready = True
                print("Models loaded: hybrid (CF + CB)")
            elif cf_path.exists() and cb_path.exists():
                self.cf = CollaborativeFilter.load(cf_path)
                self.cb = ContentBasedFilter.load(cb_path)
                self.is_ready = True
                print("Models loaded: CF + CB (no hybrid)")
            else:
                print("Warning: no trained models found. Train first with ml/scripts/train.py")
        except Exception as e:
            print(f"Warning: could not load models — {e}")

    def reload(self):
        """Hot-reload models after retraining."""
        self._load_sync()


model_store = ModelStore()
