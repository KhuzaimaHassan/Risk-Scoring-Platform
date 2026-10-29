"""
src/services/prediction_service.py
-------------------------------------
Business logic layer for fraud prediction.

This is the critical decoupling layer between the FastAPI route and the ML
inference machinery. Routes call service methods; service methods handle:
    1. Fetching transaction + dimension data from PostgreSQL
    2. Building feature vectors via FeaturePipeline
    3. Running model inference (cached model from app.state)
    4. Logging the prediction to prediction_logs
    5. Returning a typed result object

Design principles:
    - Zero direct SQLAlchemy model imports inside routes
    - All DB I/O is async (asyncpg driver)
    - Model loading: reads from caller-supplied `model` + `meta` (never
      re-loads the file on the hot path)
    - Feature computation reuses transform_from_payload() (no DB access
      inside FeaturePipeline on the inference path)
    - Predictions succeed only after their audit record commits
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    ConfusionContext,
    PredictResponse,
)
from src.database.crud.predictions import log_prediction
from src.features.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers: transaction + history DB fetch (async)
# ---------------------------------------------------------------------------

async def _fetch_scoring_context(
    db: AsyncSession,
    transaction_id: uuid.UUID,
) -> dict[str, Any] | None:
    """
    Fetch the transaction, dimensions and past-only history in one snapshot.
    Returns None if not found.
    """
    sql = text("""
        SELECT
            ft.transaction_id, ft.user_id, ft.merchant_id,
            ft.txn_timestamp, ft.amount_usd, ft.amount, ft.currency,
            ft.status, ft.channel, ft.payment_method, ft.is_international,
            ft.fraud_label,
            du.account_age_days           AS user_account_age_days,
            du.risk_tier                  AS user_risk_tier,
            du.credit_score               AS user_credit_score,
            du.kyc_verified               AS user_kyc_verified,
            du.country_code               AS user_country_code,
            dm.risk_level                 AS merchant_risk_level,
            dm.historical_fraud_rate      AS merchant_historical_fraud_rate,
            dm.is_high_risk_category      AS merchant_is_high_risk_category,
            dm.is_online_only             AS merchant_is_online_only,
            dm.avg_transaction_amount     AS merchant_avg_transaction_amount,
            dm.category                   AS merchant_category,
            dm.country_code               AS merchant_country_code,
            COALESCE(history.rows, '[]'::json) AS history_rows
        FROM   fact_transaction ft
        JOIN   dim_user     du ON du.user_id     = ft.user_id
        JOIN   dim_merchant dm ON dm.merchant_id = ft.merchant_id
        LEFT JOIN LATERAL (
            SELECT json_agg(json_build_object(
                'user_id', h.user_id, 'merchant_id', h.merchant_id,
                'txn_timestamp', h.txn_timestamp, 'amount_usd', h.amount_usd,
                'fraud_label', h.fraud_label, 'labelled_at', h.labelled_at
            ) ORDER BY h.txn_timestamp) AS rows
            FROM fact_transaction h
            WHERE (h.user_id = ft.user_id OR h.merchant_id = ft.merchant_id)
              AND h.txn_timestamp >= ft.txn_timestamp - INTERVAL '30 days'
              AND h.txn_timestamp < ft.txn_timestamp
              AND upper(h.status::text) = 'COMPLETED'
        ) history ON TRUE
        WHERE  ft.transaction_id = :txn_id
    """)
    result = await db.execute(sql, {"txn_id": str(transaction_id)})
    row = result.fetchone()
    if row is None:
        return None
    return dict(zip(result.keys(), row))


def _split_history(row):
    """Decode the shared history response without another database round trip."""
    history = pd.DataFrame(row["history_rows"], columns=[
        "user_id", "merchant_id", "txn_timestamp", "amount_usd", "fraud_label", "labelled_at",
    ])
    history["txn_timestamp"] = pd.to_datetime(history["txn_timestamp"], utc=True)
    history["amount_usd"] = pd.to_numeric(history["amount_usd"], errors="coerce")
    history["fraud_label"] = pd.to_numeric(history["fraud_label"], errors="coerce")
    return (history[history.user_id.astype(str) == str(row["user_id"])],
            history[history.merchant_id.astype(str) == str(row["merchant_id"])])



# ---------------------------------------------------------------------------
# Risk context helpers
# ---------------------------------------------------------------------------

def _build_risk_context(probability: float, is_fraud: bool, threshold: float = 0.5) -> ConfusionContext:
    """
    Map raw fraud probability to a human-readable risk band and action.

    Bands:
        [0.00, 0.25) → low      → allow
        [0.25, 0.50) → medium   → allow (flag for review if recurring)
        [0.50, 0.75) → high     → review
        [0.75, 1.00] → critical → block

    Confidence is derived from distance-to-threshold:
        |P - 0.50| < 0.10 → low confidence (borderline)
        |P - 0.50| < 0.25 → moderate
        else               → high
    """
    band = "critical" if is_fraud and probability >= max(threshold, 0.75) else "high" if is_fraud else "medium" if probability >= threshold / 2 else "low"
    action = "block" if band == "critical" else "review" if is_fraud else "allow"
    dist = abs(probability - threshold)
    confidence = "low" if dist < .1 else "moderate" if dist < .25 else "high"

    return ConfusionContext(
        risk_band=band,
        recommended_action=action,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Prediction result dataclass (internal — not Pydantic)
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Internal result returned by PredictionService before API serialisation."""
    transaction_id: uuid.UUID
    fraud_probability: float
    is_fraud: bool
    risk_context: ConfusionContext
    model_version: str
    decision_threshold: float
    log_id: uuid.UUID
    latency_ms: int
    scored_at: datetime
    feature_snapshot: dict[str, Any] | None
    timings_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PredictionService
# ---------------------------------------------------------------------------

class PredictionService:
    """
    Orchestrates the full fraud prediction lifecycle for a single transaction.

    Instantiated per-request (stateless) — the heavy model object is passed
    in at construction time from app.state (already loaded at startup).

    Attributes:
        model:         Fitted sklearn Pipeline (imputer + scaler + RF).
        model_meta:    Metadata dict from models/metadata_{version}.json.
        model_version: Version tag string.
        threshold:     Binary classification decision threshold.
        _pipeline:     Shared FeaturePipeline instance (stateless, reusable).
    """

    def __init__(
        self,
        model: Any,
        model_meta: dict[str, Any],
        model_version: str,
        threshold: float,
    ) -> None:
        self.batch_errors = []
        self.model         = model
        self.model_meta    = model_meta
        self.model_version = model_version
        self.threshold     = threshold
        self._pipeline     = FeaturePipeline(lookback_days=30)

    async def predict(
        self,
        db: AsyncSession,
        transaction_id: uuid.UUID,
        include_features: bool = False,
    ) -> PredictionResult:
        """
        Execute the full inference pipeline for a single transaction.

        Steps:
            1. Fetch transaction + dimension rows from PostgreSQL.
            2. Split the past-only 30-day history from the same database response.
            3. Build feature vector via FeaturePipeline.transform_from_payload().
            4. Run model.predict_proba() (hot path — model already in memory).
            5. Apply decision threshold → binary label.
            6. Build risk context (band + action + confidence).
            7. Commit the prediction audit record before returning success.
            8. Return PredictionResult.

        Args:
            db:             Async SQLAlchemy session.
            transaction_id: UUID of the transaction to score.
            include_features: Whether to include the feature vector in the result.

        Returns:
            PredictionResult dataclass.

        Raises:
            ValueError: If the transaction_id is not found in the database.
            RuntimeError: If feature computation or model inference fails.
        """
        t_start = time.perf_counter()
        timings = {}

        # ── Step 1: Fetch transaction + dimensions ────────────────────────
        row = await _fetch_scoring_context(db, transaction_id)
        t_transaction = time.perf_counter()
        timings["context_db"] = (t_transaction - t_start) * 1000
        if row is None:
            raise ValueError(
                f"Transaction '{transaction_id}' not found in fact_transaction. "
                f"Ensure the transaction is inserted before calling /predict."
            )

        txn_ts: datetime = row["txn_timestamp"]
        if hasattr(txn_ts, "tzinfo") and txn_ts.tzinfo is None:
            txn_ts = txn_ts.replace(tzinfo=timezone.utc)

        # ── Step 2: Split historical context from the same read snapshot ──
        user_history, merchant_history = _split_history(row)

        # ── Step 3: Build feature vector ──────────────────────────────────
        txn_dict = {
            "amount_usd":       float(row["amount_usd"]),
            "amount": float(row["amount"]),
            "is_international": bool(row["is_international"]),
            "channel":          str(row["channel"]),
            "payment_method":   str(row["payment_method"]),
            "txn_timestamp":    txn_ts,
        }
        user_dict = {
            "account_age_days": row["user_account_age_days"],
            "risk_tier":        str(row["user_risk_tier"]),
            "credit_score":     row.get("user_credit_score"),
            "kyc_verified":     bool(row["user_kyc_verified"]),
        }
        merchant_dict = {
            "risk_level":                 str(row["merchant_risk_level"]),
            "historical_fraud_rate":      float(row["merchant_historical_fraud_rate"]),
            "is_high_risk_category":      bool(row["merchant_is_high_risk_category"]),
            "is_online_only":             bool(row["merchant_is_online_only"]),
            "avg_transaction_amount":     float(row["merchant_avg_transaction_amount"]),
        }

        try:
            feature_df = await run_in_threadpool(self._pipeline.transform_from_payload,
                txn=txn_dict,
                user=user_dict,
                merchant=merchant_dict,
                user_history=user_history,
                merchant_history=merchant_history,
            )
        except Exception as exc:
            logger.error("Feature computation failed for txn %s: %s", transaction_id, exc)
            raise RuntimeError(f"Feature computation failed: {exc}") from exc

        # ── Step 4: Model inference ───────────────────────────────────────
        t_features = time.perf_counter()
        timings["features"] = (t_features - t_transaction) * 1000
        try:
            probs = await run_in_threadpool(self.model.predict_proba, feature_df)
            fraud_prob = float(probs[0, 1])
        except Exception as exc:
            logger.error("Model inference failed for txn %s: %s", transaction_id, exc)
            raise RuntimeError(f"Model inference failed: {exc}") from exc
        t_inference = time.perf_counter()
        timings["inference"] = (t_inference - t_features) * 1000

        # ── Step 5: Apply threshold ───────────────────────────────────────
        is_fraud = fraud_prob >= self.threshold
        risk_context = _build_risk_context(fraud_prob, is_fraud, self.threshold)

        # ── Step 6: Compute latency ───────────────────────────────────────
        latency_ms = int((time.perf_counter() - t_start) * 1000)
        scored_at  = datetime.now(timezone.utc)

        # ── Step 7: Commit the audit record before returning success ─────
        import json
        feature_dict = json.loads(feature_df.iloc[0].to_json())
        feature_snapshot = feature_dict if include_features else None
        log_id = await self._log_prediction_safe(
            db=db,
            transaction_id=transaction_id,
            fraud_prob=fraud_prob,
            is_fraud=is_fraud,
            latency_ms=latency_ms,
            feature_snapshot=feature_dict,  # Always log features
        )

        latency_ms = int((time.perf_counter() - t_start) * 1000)
        timings["audit"] = (time.perf_counter() - t_inference) * 1000
        timings = {stage: round(duration, 3) for stage, duration in timings.items()}
        logger.info(
            "SCORED txn=%s | prob=%.4f | label=%s | threshold=%.2f | "
            "model=%s | latency=%dms",
            transaction_id, fraud_prob, "FRAUD" if is_fraud else "LEGIT",
            self.threshold, self.model_version, latency_ms,
        )
        logger.info("SCORING_TIMINGS total_ms=%d stages_ms=%s", latency_ms, timings)

        return PredictionResult(
            transaction_id=transaction_id,
            fraud_probability=round(fraud_prob, 6),
            is_fraud=is_fraud,
            risk_context=risk_context,
            model_version=self.model_version,
            decision_threshold=self.threshold,
            log_id=log_id,
            latency_ms=latency_ms,
            scored_at=scored_at,
            feature_snapshot=feature_snapshot if include_features else None,
            timings_ms=timings,
        )

    async def predict_batch(
        self,
        db: AsyncSession,
        transaction_ids: list[uuid.UUID],
        include_features: bool = False,
    ) -> list[PredictionResult]:
        """
        Score multiple transactions sequentially.
        Skips transactions not found in the database (logs a warning per skip).
        """
        results = []
        self.batch_errors = []
        for txn_id in transaction_ids:
            try:
                results.append(await self.predict(db, txn_id, include_features))
            except Exception as exc:
                await db.rollback()
                missing = isinstance(exc, ValueError)
                self.batch_errors.append({"transaction_id": str(txn_id), "status": 404 if missing else 503,
                                          "detail": "Transaction not found" if missing else "Scoring failed; try again."})
                logger.warning("Batch item failed: %s", txn_id, exc_info=True)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _log_prediction_safe(
        self,
        db: AsyncSession,
        transaction_id: uuid.UUID,
        fraud_prob: float,
        is_fraud: bool,
        latency_ms: int,
        feature_snapshot: dict[str, Any],
    ) -> uuid.UUID:
        """
        Write prediction to prediction_logs table.
        On any failure: roll back and raise; no success without a durable log.
        """
        request_id = uuid.uuid4()
        try:
            log_entry = await log_prediction(
                db=db,
                request_id=request_id,
                transaction_id=transaction_id,
                model_name=self.model_meta.get("model_name", "fraud_classifier"),
                model_version=self.model_version,
                fraud_score=fraud_prob,
                risk_label=int(is_fraud),
                decision_threshold=self.threshold,
                feature_vector=feature_snapshot,
                latency_ms=latency_ms,
            )
            await db.commit()
            return log_entry.log_id
        except Exception as exc:
            await db.rollback()
            logger.error("Audit write failed for transaction %s", transaction_id, exc_info=True)
            raise RuntimeError("Prediction could not be durably recorded") from exc
