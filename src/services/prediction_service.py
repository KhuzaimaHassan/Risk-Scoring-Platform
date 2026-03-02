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
    - Prediction logging is a fire-and-forget async operation — failures
      are logged as warnings but do NOT raise errors to the client
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text
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

async def _fetch_transaction_with_dims(
    db: AsyncSession,
    transaction_id: uuid.UUID,
) -> dict[str, Any] | None:
    """
    Fetch a single transaction row joined with dim_user and dim_merchant.
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
            dm.country_code               AS merchant_country_code
        FROM   fact_transaction ft
        JOIN   dim_user     du ON du.user_id     = ft.user_id
        JOIN   dim_merchant dm ON dm.merchant_id = ft.merchant_id
        WHERE  ft.transaction_id = :txn_id
    """)
    result = await db.execute(sql, {"txn_id": str(transaction_id)})
    row = result.fetchone()
    if row is None:
        return None
    return dict(zip(result.keys(), row))


async def _fetch_user_history(
    db: AsyncSession,
    user_id: uuid.UUID,
    before_ts: datetime,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Fetch historical user transactions strictly before before_ts."""
    from datetime import timedelta

    lower = before_ts - timedelta(days=lookback_days)
    sql = text("""
        SELECT txn_timestamp, amount_usd, fraud_label
        FROM   fact_transaction
        WHERE  user_id = :uid
          AND  txn_timestamp < :upper
          AND  txn_timestamp >= :lower
          AND  status = 'completed'
        ORDER  BY txn_timestamp ASC
    """)
    result = await db.execute(sql, {
        "uid": str(user_id),
        "upper": before_ts,
        "lower": lower,
    })
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame(columns=["txn_timestamp", "amount_usd", "fraud_label"])
    df = pd.DataFrame(rows, columns=["txn_timestamp", "amount_usd", "fraud_label"])
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    return df


async def _fetch_merchant_history(
    db: AsyncSession,
    merchant_id: uuid.UUID,
    before_ts: datetime,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Fetch historical merchant transactions strictly before before_ts."""
    from datetime import timedelta

    lower = before_ts - timedelta(days=lookback_days)
    sql = text("""
        SELECT txn_timestamp, amount_usd, fraud_label
        FROM   fact_transaction
        WHERE  merchant_id = :mid
          AND  txn_timestamp < :upper
          AND  txn_timestamp >= :lower
          AND  status = 'completed'
        ORDER  BY txn_timestamp ASC
    """)
    result = await db.execute(sql, {
        "mid": str(merchant_id),
        "upper": before_ts,
        "lower": lower,
    })
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame(columns=["txn_timestamp", "amount_usd", "fraud_label"])
    df = pd.DataFrame(rows, columns=["txn_timestamp", "amount_usd", "fraud_label"])
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Risk context helpers
# ---------------------------------------------------------------------------

def _build_risk_context(probability: float, is_fraud: bool) -> ConfusionContext:
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
    if probability < 0.25:
        band, action = "low", "allow"
    elif probability < 0.50:
        band, action = "medium", "allow"
    elif probability < 0.75:
        band, action = "high", "review"
    else:
        band, action = "critical", "block"

    dist = abs(probability - 0.50)
    if dist < 0.10:
        confidence = "low"
    elif dist < 0.25:
        confidence = "moderate"
    else:
        confidence = "high"

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
        self.model         = model
        self.model_meta    = model_meta
        self.model_version = model_version
        self.threshold     = threshold
        self._pipeline     = FeaturePipeline(lookback_days=7)

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
            2. Fetch user and merchant history (async, last 7 days).
            3. Build feature vector via FeaturePipeline.transform_from_payload().
            4. Run model.predict_proba() (hot path — model already in memory).
            5. Apply decision threshold → binary label.
            6. Build risk context (band + action + confidence).
            7. Log prediction to prediction_logs (async, non-blocking on failure).
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

        # ── Step 1: Fetch transaction + dimensions ────────────────────────
        row = await _fetch_transaction_with_dims(db, transaction_id)
        if row is None:
            raise ValueError(
                f"Transaction '{transaction_id}' not found in fact_transaction. "
                f"Ensure the transaction is inserted before calling /predict."
            )

        txn_ts: datetime = row["txn_timestamp"]
        if hasattr(txn_ts, "tzinfo") and txn_ts.tzinfo is None:
            txn_ts = txn_ts.replace(tzinfo=timezone.utc)

        user_id    = uuid.UUID(str(row["user_id"]))
        merchant_id = uuid.UUID(str(row["merchant_id"]))

        # ── Step 2: Fetch historical context (async, past-only) ──────────
        user_history, merchant_history = await _parallel_history_fetch(
            db, user_id, merchant_id, txn_ts
        )

        # ── Step 3: Build feature vector ──────────────────────────────────
        txn_dict = {
            "amount_usd":       float(row["amount_usd"]),
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
            feature_df = self._pipeline.transform_from_payload(
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
        try:
            fraud_prob = float(self.model.predict_proba(feature_df)[0, 1])
        except Exception as exc:
            logger.error("Model inference failed for txn %s: %s", transaction_id, exc)
            raise RuntimeError(f"Model inference failed: {exc}") from exc

        # ── Step 5: Apply threshold ───────────────────────────────────────
        is_fraud = fraud_prob >= self.threshold
        risk_context = _build_risk_context(fraud_prob, is_fraud)

        # ── Step 6: Compute latency ───────────────────────────────────────
        latency_ms = int((time.perf_counter() - t_start) * 1000)
        scored_at  = datetime.now(timezone.utc)

        # ── Step 7: Log prediction (best-effort — failures are logged) ────
        feature_snapshot = feature_df.iloc[0].to_dict() if include_features else None
        log_id = await self._log_prediction_safe(
            db=db,
            transaction_id=transaction_id,
            fraud_prob=fraud_prob,
            is_fraud=is_fraud,
            latency_ms=latency_ms,
            feature_snapshot=feature_df.iloc[0].to_dict(),  # Always log features
        )

        logger.info(
            "SCORED txn=%s | prob=%.4f | label=%s | threshold=%.2f | "
            "model=%s | latency=%dms",
            transaction_id, fraud_prob, "FRAUD" if is_fraud else "LEGIT",
            self.threshold, self.model_version, latency_ms,
        )

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
        for txn_id in transaction_ids:
            try:
                result = await self.predict(db, txn_id, include_features)
                results.append(result)
            except ValueError as exc:
                logger.warning("Skipping txn %s in batch: %s", txn_id, exc)
            except Exception as exc:
                logger.error("Batch item failed for txn %s: %s", txn_id, exc)
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
        On any failure: log a warning and return a new UUID (non-fatal).
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
            return log_entry.log_id
        except Exception as exc:
            logger.warning(
                "Prediction log write failed for txn %s: %s — "
                "prediction was returned successfully to the client.",
                transaction_id, exc
            )
            return request_id


async def _parallel_history_fetch(
    db: AsyncSession,
    user_id: uuid.UUID,
    merchant_id: uuid.UUID,
    txn_ts: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concurrently fetch user and merchant history.
    Running both queries takes roughly max(t1, t2) instead of t1 + t2.
    """
    import asyncio
    user_hist, merchant_hist = await asyncio.gather(
        _fetch_user_history(db, user_id, txn_ts),
        _fetch_merchant_history(db, merchant_id, txn_ts),
    )
    return user_hist, merchant_hist
