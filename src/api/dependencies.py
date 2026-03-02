"""
src/api/dependencies.py
------------------------
FastAPI dependency injection: DB sessions, model loader, and service factory.

─────────────────────────────────────────────────────────────────────────────
Model Caching Strategy
─────────────────────────────────────────────────────────────────────────────
The model is loaded ONCE at application startup via the FastAPI lifespan
handler (defined in main.py). The fitted sklearn Pipeline and its metadata
are stored in `request.app.state`:

    app.state.model          — fitted sklearn Pipeline (thread-safe for reads)
    app.state.model_meta     — metadata dict from models/metadata_{v}.json
    app.state.model_version  — version tag string
    app.state.threshold      — decision threshold float

Dependencies here read from app.state — they do NOT call ModelRegistry or
load the file system on hot paths. The model is effectively a singleton for
the lifetime of the process.

Reload strategy for production:
    Option A (zero-downtime): Rolling restart via k8s/docker with new image.
    Option B (hot-swap): POST /model-info/reload endpoint (not in MVP) that
                         re-reads app.state under a write lock.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import get_async_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database session dependency (async)
# ---------------------------------------------------------------------------

AsyncDBSession = Annotated[AsyncSession, Depends(get_async_db)]
"""
Type alias for FastAPI routes that need an async DB session.

Usage in routes:
    @router.post("/predict")
    async def predict(db: AsyncDBSession, ...):
        ...
"""


# ---------------------------------------------------------------------------
# Model state accessors
# ---------------------------------------------------------------------------

def get_model(request: Request) -> Any:
    """
    Return the cached fitted sklearn Pipeline from application state.

    Raises HTTP 503 if the model has not been loaded (startup failure).
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        logger.error("Model is not loaded — app.state.model is None.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. The service is still starting up or failed to load.",
        )
    return model


def get_model_meta(request: Request) -> dict[str, Any]:
    """
    Return the model metadata dict from application state.

    Raises HTTP 503 if the metadata has not been loaded.
    """
    meta = getattr(request.app.state, "model_meta", None)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not available.",
        )
    return meta


def get_model_version(request: Request) -> str:
    """Return the currently loaded model version tag."""
    return getattr(request.app.state, "model_version", "unknown")


def get_decision_threshold(request: Request) -> float:
    """Return the decision threshold for the currently loaded model."""
    return getattr(request.app.state, "threshold", 0.50)


# Annotated type aliases for route signatures
CachedModel        = Annotated[Any,             Depends(get_model)]
CachedModelMeta    = Annotated[dict[str, Any],  Depends(get_model_meta)]
CachedModelVersion = Annotated[str,             Depends(get_model_version)]
CachedThreshold    = Annotated[float,           Depends(get_decision_threshold)]


# ---------------------------------------------------------------------------
# DB connectivity check (used by /health)
# ---------------------------------------------------------------------------

async def check_db_connection(db: AsyncDBSession) -> bool:
    """
    Light-weight DB connectivity probe: runs `SELECT 1`.
    Returns True on success, False on any exception.
    """
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return False
