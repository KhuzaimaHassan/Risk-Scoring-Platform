"""
src/api/routes/predict.py
--------------------------
Prediction endpoint routes.

Routes:
    POST /predict         — Score a single transaction
    POST /predict/batch   — Score up to 200 transactions
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status

from src.api.dependencies import AsyncDBSession, CachedThreshold
from src.api.schemas import (
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from src.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Inference"])


def _get_prediction_service(request: Request, threshold: float) -> PredictionService:
    """Build a PredictionService from application state (model cached at startup)."""
    model      = request.app.state.model
    model_meta = request.app.state.model_meta
    version    = request.app.state.model_version
    return PredictionService(
        model=model,
        model_meta=model_meta,
        model_version=version,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# POST /predict — Single transaction scoring
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a transaction for fraud",
    description=(
        "Accepts a transaction_id that already exists in fact_transaction, "
        "fetches historical context, generates features, runs the fraud model, "
        "logs the prediction, and returns a risk score with contextual metadata."
    ),
    responses={
        404: {"description": "Transaction not found"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_transaction(
    payload: PredictRequest,
    request: Request,
    db: AsyncDBSession,
    threshold: CachedThreshold,
) -> PredictResponse:
    """
    Full inference pipeline for a single transaction.

    Request lifecycle:
        1. Pydantic validates the payload (schema layer).
        2. PredictionService fetches transaction + history from DB.
        3. FeaturePipeline builds the feature vector (strictly past-only).
        4. Cached sklearn Pipeline runs predict_proba().
        5. Prediction is logged to prediction_logs (best-effort).
        6. PredictResponse is returned.
    """
    service = _get_prediction_service(request, threshold)

    try:
        result = await service.predict(
            db=db,
            transaction_id=payload.transaction_id,
            include_features=payload.include_features,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Inference pipeline error for txn %s: %s", payload.transaction_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    return PredictResponse(
        transaction_id=result.transaction_id,
        fraud_probability=round(result.fraud_probability, 6),
        is_fraud=result.is_fraud,
        risk_score=round(result.fraud_probability * 100, 2),
        decision_threshold=result.decision_threshold,
        model_version=result.model_version,
        risk_context=result.risk_context,
        log_id=result.log_id,
        latency_ms=result.latency_ms,
        scored_at=result.scored_at,
        feature_snapshot=result.feature_snapshot,
    )


# ---------------------------------------------------------------------------
# POST /predict/batch — Batch scoring
# ---------------------------------------------------------------------------

@router.post(
    "/batch",
    response_model=PredictBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Score multiple transactions",
    description=(
        "Score up to 200 transactions in a single call. "
        "Transactions not found in the database are skipped (warnings logged)."
    ),
)
async def predict_batch(
    payload: PredictBatchRequest,
    request: Request,
    db: AsyncDBSession,
    threshold: CachedThreshold,
) -> PredictBatchResponse:
    """Batch prediction endpoint — sequential scoring with per-item error isolation."""
    import time
    t0 = time.perf_counter()

    service = _get_prediction_service(request, threshold)
    results = await service.predict_batch(
        db=db,
        transaction_ids=[uuid.UUID(str(tid)) for tid in payload.transaction_ids],
        include_features=payload.include_features,
    )

    batch_ms = int((time.perf_counter() - t0) * 1000)

    response_items = [
        PredictResponse(
            transaction_id=r.transaction_id,
            fraud_probability=round(r.fraud_probability, 6),
            is_fraud=r.is_fraud,
            risk_score=round(r.fraud_probability * 100, 2),
            decision_threshold=r.decision_threshold,
            model_version=r.model_version,
            risk_context=r.risk_context,
            log_id=r.log_id,
            latency_ms=r.latency_ms,
            scored_at=r.scored_at,
            feature_snapshot=r.feature_snapshot,
        )
        for r in results
    ]

    return PredictBatchResponse(
        results=response_items,
        total_requested=len(payload.transaction_ids),
        total_scored=len(results),
        total_fraud=sum(1 for r in results if r.is_fraud),
        batch_latency_ms=batch_ms,
    )
