"""
src/api/routes/health.py
--------------------------
System health and model info endpoints.

Routes:
    GET /health        — API liveness + DB + model status
    GET /model-info    — Current production model metadata + metrics
    GET /models        — Full model registry listing
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request, status

from src.api.dependencies import AsyncDBSession, check_db_connection
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ModelListItem,
    ModelListResponse,
    ModelMetrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Operations"])


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
    description=(
        "Returns the operational status of the API including model load state, "
        "database connectivity, and service uptime."
    ),
)
async def health_check(request: Request, db: AsyncDBSession) -> HealthResponse:
    """
    Liveness + readiness probe combined.
    Used by Docker health checks and Kubernetes liveness/readiness probes.
    """
    model_loaded  = getattr(request.app.state, "model", None) is not None
    model_version = getattr(request.app.state, "model_version", None)
    start_time    = getattr(request.app.state, "start_time", time.time())
    uptime        = time.time() - start_time

    db_ok = await check_db_connection(db)

    overall_status = "ok" if (model_loaded and db_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        model_loaded=model_loaded,
        model_version=model_version,
        db_connected=db_ok,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Current production model information",
    description=(
        "Returns the version, metrics, hyperparameters, and feature list "
        "for the model currently serving predictions."
    ),
)
async def model_info(request: Request) -> ModelInfoResponse:
    """
    Return metadata for the currently loaded model.
    No DB I/O — reads from app.state which is populated at startup.
    """
    meta = getattr(request.app.state, "model_meta", None)
    if meta is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not available. Service may still be starting up.",
        )

    raw_metrics  = meta.get("metrics", {})
    hyperparams  = meta.get("hyperparameters", {})
    feature_cols = meta.get("feature_columns", [])

    return ModelInfoResponse(
        model_name=meta.get("model_name", meta.get("name", "fraud_classifier")),
        model_version=meta.get("version", "unknown"),
        stage="production",
        trained_at=meta.get("trained_at"),
        n_features=meta.get("n_features", len(feature_cols)),
        feature_columns=feature_cols,
        hyperparameters=hyperparams,
        metrics=ModelMetrics(
            roc_auc=raw_metrics.get("roc_auc"),
            pr_auc=raw_metrics.get("pr_auc"),
            f1=raw_metrics.get("f1"),
            precision=raw_metrics.get("precision"),
            recall=raw_metrics.get("recall"),
            threshold_used=raw_metrics.get("threshold_used"),
            training_rows=meta.get("training_rows"),
        ),
        artifact_path=meta.get("artifact_path", ""),
    )


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get(
    "/models",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all model versions",
    description="Returns all versions in the model registry with their lifecycle stages.",
)
async def list_models(request: Request) -> ModelListResponse:
    """Load registry from disk and return version list."""
    from pathlib import Path
    from src.training.model_registry import ModelRegistry

    # Resolve model dir from app settings or default
    model_dir = getattr(request.app.state, "model_dir", Path("models"))
    registry  = ModelRegistry(base_dir=model_dir)
    summary   = registry.summary()
    versions  = registry.list_models()
    current   = summary.get("current_production_model")

    items = [
        ModelListItem(
            version=v["version"],
            stage=v.get("stage", "unknown"),
            trained_at=v.get("trained_at"),
            roc_auc=v.get("metrics", {}).get("roc_auc"),
            f1=v.get("metrics", {}).get("f1"),
            is_production=(v["version"] == current),
        )
        for v in versions
    ]

    return ModelListResponse(
        current_production=current,
        total_versions=summary["total_versions"],
        versions=items,
    )
