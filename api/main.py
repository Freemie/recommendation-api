from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.cache import init_redis, close_redis
from core.model_store import model_store
from routers import auth, recommendations, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_redis()
    await model_store.load()
    yield
    # Shutdown
    await close_redis()


app = FastAPI(
    title="Recommendation System API",
    description="Movie recommendations via collaborative filtering, content-based, and hybrid models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"Resource not found: {exc}"},
    )


app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(recommendations.router, prefix="", tags=["Recommendations"])
app.include_router(feedback.router, prefix="", tags=["Feedback"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "models_loaded": model_store.is_ready}
