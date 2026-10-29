"""
src/features/feature_pipeline.py
-----------------------------------
Central orchestration layer for feature engineering.

This module is the single entry point for feature computation in
both TRAINING and INFERENCE modes. It owns all database I/O,
coordinates feature_extraction.py and feature_aggregation.py,
and returns a clean, consistently-ordered pandas DataFrame.

─────────────────────────────────────────────────────────────────
How Training Calls This Pipeline
─────────────────────────────────────────────────────────────────
    from src.features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    X, y = pipeline.build_training_dataset(since=datetime(2025, 9, 1))
    # X: pd.DataFrame shape (n_samples, n_features) — all FEATURE_COLUMNS
    # y: pd.Series[int]  shape (n_samples,)          — fraud_label

─────────────────────────────────────────────────────────────────
How Inference Calls This Pipeline
─────────────────────────────────────────────────────────────────
    from src.features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline()

    # Single transaction (real-time API path)
    feature_row = pipeline.transform_single(transaction_id="<uuid>")
    # feature_row: pd.DataFrame shape (1, n_features)

    # Batch scoring
    feature_df = pipeline.transform_batch(transaction_ids=["<uuid1>", "<uuid2>"])
    # feature_df: pd.DataFrame shape (n, n_features)

─────────────────────────────────────────────────────────────────
Data Leakage Contract
─────────────────────────────────────────────────────────────────
    For every transaction T being featurised:
        - Only transactions with txn_timestamp < T.txn_timestamp are
          passed to the aggregation layer.
        - This strict less-than guarantees no future information leaks
          into the feature vector, even during batch training.
        - The same logic applies in inference, where the current
          transaction has not yet been committed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import get_settings
from src.database.session import sync_db_context
from src.features.feature_aggregation import (
    AGGREGATED_FEATURE_NAMES,
    compute_all_aggregated_features,
)
from src.features.feature_extraction import (
    RAW_FEATURE_NAMES,
    extract_raw_features,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Feature column contract
# ---------------------------------------------------------------------------

#: The canonical ordered list of all feature columns in the output DataFrame.
#: Training and inference MUST always return columns in this exact order.
#: The model registry stores this list alongside model artifacts.
FEATURE_SCHEMA_VERSION = 2
# Profile snapshots are not stored, so mutable profile fields cannot be used as-of an event.
_EXCLUDED = {"user_account_age_days", "user_risk_tier_code", "user_credit_score", "user_kyc_verified", "merchant_risk_level_code"}
FEATURE_COLUMNS: list[str] = [c for c in RAW_FEATURE_NAMES if c not in _EXCLUDED] + AGGREGATED_FEATURE_NAMES

#: Target label column name for training
TARGET_COLUMN: str = "fraud_label"

#: Fill values for imputing missing features (used as a last-resort safety net
#: after all extraction logic runs). Tree-based models tolerate these fills well.
_IMPUTATION_DEFAULTS: dict[str, float] = {
    "amount_usd": 0.0,
    "is_international": 0,
    "is_online_merchant": 0,
    "hour_of_day": 12,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_night": 0,
    "channel_code": 0,
    "payment_method_code": 0,
    "user_account_age_days": 365,
    "user_risk_tier_code": 1,
    "user_credit_score": 670.0,
    "user_kyc_verified": 0,
    "merchant_risk_level_code": 1,
    "merchant_historical_fraud_rate": 0.0,
    "merchant_is_high_risk_category": 0,
    "merchant_avg_transaction_amount": 100.0,
    "user_txn_count_1h": 0,
    "user_txn_count_24h": 0,
    "user_txn_count_7d": 0,
    "user_avg_amount_7d": 0.0,
    "user_std_amount_7d": 0.0,
    "user_max_amount_7d": 0.0,
    "user_total_amount_24h": 0.0,
    "user_velocity_ratio_1h_24h": 0.0,
    "time_since_last_txn_minutes": 43200.0,
    "amount_vs_user_avg_ratio": 1.0,
    "amount_vs_merchant_avg_ratio": 1.0,
    "merchant_txn_count_7d": 0,
    "merchant_fraud_rate_7d": 0.0,
    "merchant_avg_amount_7d": 0.0,
}


# ---------------------------------------------------------------------------
# SQL helpers — isolated here so aggregation / extraction remain DB-free
# ---------------------------------------------------------------------------

def _load_full_transaction_table(
    db_session: Any,
    since: datetime | None = None,
    labelled_only: bool = True,
) -> pd.DataFrame:
    """
    Load the full fact_transaction table joined with dim_user and dim_merchant.

    Pulls only the columns needed for feature calculation — avoids loading
    PII fields (full_name, email_hash) or heavy text columns (review_notes).

    Args:
        db_session:    Sync SQLAlchemy session.
        since:         Optional lower bound on txn_timestamp.
        labelled_only: If True (training mode), only return rows where
                       fraud_label IS NOT NULL.

    Returns:
        pd.DataFrame with one row per transaction, columns covering all
        fields needed by feature_extraction and feature_aggregation.
    """
    label_clause = "AND ft.fraud_label IS NOT NULL" if labelled_only else ""
    since_clause = f"AND ft.txn_timestamp >= :since" if since else ""

    sql = text(f"""
        SELECT
            ft.transaction_id,
            ft.user_id,
            ft.merchant_id,
            ft.txn_timestamp,
            ft.amount_usd,
            ft.amount,
            ft.currency,
            ft.status,
            ft.channel,
            ft.payment_method,
            ft.is_international,
            ft.fraud_label, ft.labelled_at,
            -- User dimension (snapshot at query time)
            du.account_age_days            AS user_account_age_days,
            du.risk_tier                   AS user_risk_tier,
            du.credit_score                AS user_credit_score,
            du.kyc_verified                AS user_kyc_verified,
            du.country_code                AS user_country_code,
            -- Merchant dimension
            dm.risk_level                  AS merchant_risk_level,
            dm.historical_fraud_rate       AS merchant_historical_fraud_rate,
            dm.is_high_risk_category       AS merchant_is_high_risk_category,
            dm.is_online_only              AS merchant_is_online_only,
            dm.avg_transaction_amount      AS merchant_avg_transaction_amount,
            dm.category                    AS merchant_category,
            dm.country_code                AS merchant_country_code
        FROM   fact_transaction ft
        JOIN   dim_user     du ON du.user_id     = ft.user_id
        JOIN   dim_merchant dm ON dm.merchant_id = ft.merchant_id
        WHERE  upper(ft.status::text) = 'COMPLETED'
               {label_clause}
               {since_clause}
        ORDER  BY ft.txn_timestamp ASC
    """)

    params: dict[str, Any] = {}
    if since:
        params["since"] = since

    result = db_session.execute(sql, params)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Cast Decimal columns returned by psycopg2 for NUMERIC/FLOAT DB types to float.
    # PostgreSQL NUMERIC columns come back as decimal.Decimal — pandas/numpy ops fail.
    _NUMERIC_COLS = [
        "amount_usd", "amount",
        "user_account_age_days", "user_credit_score",
        "merchant_historical_fraud_rate", "merchant_avg_transaction_amount",
        "fraud_label",
    ]
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure timestamp column is tz-aware UTC
    if not df.empty and "txn_timestamp" in df.columns:
        df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)

    logger.debug("Loaded %d transactions from DB.", len(df))
    return df



def _get_single_transaction(
    db_session: Any,
    transaction_id: str | uuid.UUID,
) -> pd.Series | None:
    """
    Load a single transaction with its dimension attributes.

    Returns None if transaction_id is not found.
    """
    sql = text("""
        SELECT
            ft.transaction_id, ft.user_id, ft.merchant_id,
            ft.txn_timestamp, ft.amount_usd, ft.amount, ft.currency,
            ft.status, ft.channel, ft.payment_method, ft.is_international,
            ft.fraud_label, ft.labelled_at,
            du.account_age_days            AS user_account_age_days,
            du.risk_tier                   AS user_risk_tier,
            du.credit_score                AS user_credit_score,
            du.kyc_verified                AS user_kyc_verified,
            du.country_code                AS user_country_code,
            dm.risk_level                  AS merchant_risk_level,
            dm.historical_fraud_rate       AS merchant_historical_fraud_rate,
            dm.is_high_risk_category       AS merchant_is_high_risk_category,
            dm.is_online_only              AS merchant_is_online_only,
            dm.avg_transaction_amount      AS merchant_avg_transaction_amount,
            dm.category                    AS merchant_category,
            dm.country_code                AS merchant_country_code
        FROM   fact_transaction ft
        JOIN   dim_user     du ON du.user_id     = ft.user_id
        JOIN   dim_merchant dm ON dm.merchant_id = ft.merchant_id
        WHERE  ft.transaction_id = :txn_id
    """)
    result = db_session.execute(sql, {"txn_id": str(transaction_id)})
    row = result.fetchone()
    if row is None:
        return None
    s = pd.Series(dict(zip(result.keys(), row)))

    # Ensure timestamp is tz-aware UTC (handle both naive and tz-aware returns)
    _ts = pd.Timestamp(s["txn_timestamp"])
    if _ts.tzinfo is None:
        _ts = _ts.tz_localize("UTC")
    else:
        _ts = _ts.tz_convert("UTC")
    s["txn_timestamp"] = _ts

    # Cast NUMERIC/DECIMAL columns from psycopg2 to Python float
    for _col in ["amount_usd", "amount", "user_credit_score",
                 "merchant_historical_fraud_rate", "merchant_avg_transaction_amount"]:
        if _col in s.index and s[_col] is not None:
            try:
                s[_col] = float(s[_col])
            except (TypeError, ValueError):
                pass
    return s




def _get_user_history(
    db_session: Any,
    user_id: str | uuid.UUID,
    before_ts: datetime,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch historical transactions for a given user strictly before `before_ts`.

    Args:
        user_id:      UUID of the user.
        before_ts:    Exclusive upper bound — only rows with timestamp < before_ts.
        lookback_days: How far back to load (trade-off: coverage vs. query cost).

    Returns:
        DataFrame with txn_timestamp and amount_usd columns.
    """
    cutoff_lower = before_ts - timedelta(days=lookback_days)
    sql = text("""
        SELECT txn_timestamp, amount_usd, fraud_label, labelled_at
        FROM   fact_transaction
        WHERE  user_id  = :user_id
          AND  txn_timestamp < :upper
          AND  txn_timestamp >= :lower
          AND  upper(status::text) = 'COMPLETED'
        ORDER  BY txn_timestamp ASC
    """)
    result = db_session.execute(sql, {
        "user_id": str(user_id),
        "upper": before_ts,
        "lower": cutoff_lower,
    })
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    if not df.empty:
        df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
        df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    return df


def _get_merchant_history(
    db_session: Any,
    merchant_id: str | uuid.UUID,
    before_ts: datetime,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch historical transactions for a given merchant strictly before `before_ts`.
    """
    cutoff_lower = before_ts - timedelta(days=lookback_days)
    sql = text("""
        SELECT txn_timestamp, amount_usd, fraud_label, labelled_at
        FROM   fact_transaction
        WHERE  merchant_id = :merchant_id
          AND  txn_timestamp < :upper
          AND  txn_timestamp >= :lower
          AND  upper(status::text) = 'COMPLETED'
        ORDER  BY txn_timestamp ASC
    """)
    result = db_session.execute(sql, {
        "merchant_id": str(merchant_id),
        "upper": before_ts,
        "lower": cutoff_lower,
    })
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    if not df.empty:
        df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
        df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _featurise_single_row(
    txn_row: pd.Series,
    user_history: pd.DataFrame,
    merchant_history: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build the complete feature dict for a single transaction row.
    Bridges feature_extraction and feature_aggregation with the correct
    dict shapes each module expects.

    Args:
        txn_row:          One row as a pandas Series (from the joined query).
        user_history:     Past transactions for this user (before txn_row).
        merchant_history: Past transactions at this merchant (before txn_row).

    Returns:
        Flat dict with all keys from FEATURE_COLUMNS.
    """
    # Convert row to dicts in the shape each extractor expects
    txn_dict: dict[str, Any] = {
        "amount_usd": txn_row["amount_usd"],
        "amount": txn_row.get("amount", txn_row["amount_usd"]),
        "is_international": txn_row["is_international"],
        "channel": txn_row["channel"],
        "payment_method": txn_row["payment_method"],
        "txn_timestamp": txn_row["txn_timestamp"],
    }
    user_dict: dict[str, Any] = {
        "account_age_days": txn_row["user_account_age_days"],
        "risk_tier": txn_row["user_risk_tier"],
        "credit_score": txn_row.get("user_credit_score"),
        "kyc_verified": txn_row["user_kyc_verified"],
    }
    agg_feats = compute_all_aggregated_features(
        user_history, merchant_history, txn_row["txn_timestamp"],
        float(txn_row["amount_usd"]), static_merchant_avg=0.0,
    )
    merchant_dict: dict[str, Any] = {
        "risk_level": txn_row["merchant_risk_level"],
        "historical_fraud_rate": agg_feats["merchant_fraud_rate_7d"],
        "is_high_risk_category": txn_row["merchant_is_high_risk_category"],
        "is_online_only": txn_row["merchant_is_online_only"],
        "avg_transaction_amount": agg_feats["merchant_avg_amount_7d"],
    }

    # 1. Raw features (no historical lookback)
    raw_feats = extract_raw_features(txn_dict, user_dict, merchant_dict)

    # 2. Aggregated rolling features (strictly historical)
    return {**raw_feats, **agg_feats}


def _apply_final_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safety-net imputation applied AFTER all feature logic runs.
    Fills any remaining NaN values with domain-appropriate defaults.

    This should rarely trigger — it is a defensive last step to ensure
    the output DataFrame is always fully numeric with no NaN values.
    """
    for col, default in _IMPUTATION_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # Any column not in _IMPUTATION_DEFAULTS gets filled with 0.0
    remaining_na_cols = df.columns[df.isna().any()].tolist()
    if remaining_na_cols:
        logger.warning(
            "Unexpected NaN in columns after imputation: %s — filling with 0.0",
            remaining_na_cols,
        )
        df[remaining_na_cols] = df[remaining_na_cols].fillna(0.0)

    return df


def _enforce_column_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that the DataFrame always has exactly FEATURE_COLUMNS in
    the correct order. Missing columns get filled with defaults; extra
    columns are silently dropped.
    """
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = _IMPUTATION_DEFAULTS.get(col, 0.0)
    return df[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# FeaturePipeline — the public interface
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    A stateless feature pipeline that computes the complete feature matrix
    for both training and real-time inference.

    Attributes:
        lookback_days: Maximum lookback window for historical aggregations.
                       Must be >= 7 (longest rolling window in aggregation).
    """

    def __init__(self, lookback_days: int = 30) -> None:
        if lookback_days < 30:
            raise ValueError("lookback_days must be >= 30 to cover the longest rolling window.")
        self.lookback_days = lookback_days
        logger.info("FeaturePipeline initialised (lookback_days=%d).", lookback_days)

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------

    def build_training_dataset(self, since=None, until=None):
        # Keep unlabelled transactions as behavioural context. Only targets are filtered.
        history_start = since - timedelta(days=self.lookback_days) if since else None
        with sync_db_context() as db:
            df = _load_full_transaction_table(db, since=history_start, labelled_only=False)
        if df.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series([], dtype=int)
        df = df.sort_values("txn_timestamp", kind="stable").reset_index(drop=True)
        cutoff = pd.Timestamp(until) if until else pd.Timestamp.now(tz="UTC")
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        df = df[df.txn_timestamp < cutoff].copy()
        users = {k: g for k, g in df.groupby("user_id", sort=False)}
        merchants = {k: g for k, g in df.groupby("merchant_id", sort=False)}
        def window(group, ts):
            stamps = pd.DatetimeIndex(group.txn_timestamp)
            start = stamps.searchsorted(ts - timedelta(days=self.lookback_days), side="left")
            end = stamps.searchsorted(ts, side="left")
            return group.iloc[start:end]
        features, labels = [], []
        for _, row in df.iterrows():
            if pd.isna(row.fraud_label) or (since and row.txn_timestamp < pd.Timestamp(since)):
                continue
            labelled_at = row.get("labelled_at")
            if pd.isna(labelled_at) or pd.Timestamp(labelled_at) >= cutoff:
                continue
            features.append(_featurise_single_row(row, window(users[row.user_id], row.txn_timestamp),
                                                   window(merchants[row.merchant_id], row.txn_timestamp)))
            labels.append(int(row.fraud_label))
        X = _enforce_column_order(_apply_final_imputation(pd.DataFrame(features)))
        return X, pd.Series(labels, dtype=int, name=TARGET_COLUMN)

    def transform_single(
        self,
        transaction_id: str | uuid.UUID,
    ) -> pd.DataFrame:
        """
        Compute features for a single transaction — the real-time API path.

        Queries historic data from DB for the rolling window at the moment
        of THIS transaction's timestamp.

        Args:
            transaction_id: UUID of the transaction to score.

        Returns:
            pd.DataFrame with shape (1, len(FEATURE_COLUMNS)).

        Raises:
            ValueError: If the transaction_id is not found in the database.
        """
        with sync_db_context() as db:
            txn_row = _get_single_transaction(db, transaction_id)
            if txn_row is None:
                raise ValueError(f"Transaction not found: {transaction_id}")

            ts       = txn_row["txn_timestamp"]
            user_id  = txn_row["user_id"]
            merch_id = txn_row["merchant_id"]

            user_history     = _get_user_history(db, user_id, ts, self.lookback_days)
            merchant_history = _get_merchant_history(db, merch_id, ts, self.lookback_days)

        features = _featurise_single_row(txn_row, user_history, merchant_history)
        df = pd.DataFrame([features])
        df = _apply_final_imputation(df)
        df = _enforce_column_order(df)
        return df

    # ------------------------------------------------------------------
    # Inference mode — batch transactions
    # ------------------------------------------------------------------

    def transform_batch(
        self,
        transaction_ids: Sequence[str | uuid.UUID],
    ) -> pd.DataFrame:
        """
        Compute features for a batch of transactions — used for batch scoring
        or backfilling prediction logs.

        Each transaction is processed independently: its rolling window
        uses only transactions before its own timestamp, so there is no
        cross-contamination within the batch.

        Args:
            transaction_ids: List/tuple of transaction UUIDs to score.

        Returns:
            pd.DataFrame with shape (len(transaction_ids), len(FEATURE_COLUMNS)).
            Rows match the input order. Transactions not found are skipped
            (a WARNING is logged for each missing ID).
        """
        logger.info("Batch transform: %d transaction IDs.", len(transaction_ids))

        feature_rows: list[dict[str, Any]] = []
        found_ids: list[str] = []

        with sync_db_context() as db:
            for txn_id in transaction_ids:
                txn_row = _get_single_transaction(db, txn_id)
                if txn_row is None:
                    logger.warning("Transaction not found in batch: %s — skipping.", txn_id)
                    continue

                ts       = txn_row["txn_timestamp"]
                user_id  = txn_row["user_id"]
                merch_id = txn_row["merchant_id"]

                user_history     = _get_user_history(db, user_id, ts, self.lookback_days)
                merchant_history = _get_merchant_history(db, merch_id, ts, self.lookback_days)

                features = _featurise_single_row(txn_row, user_history, merchant_history)
                feature_rows.append(features)
                found_ids.append(str(txn_id))

        if not feature_rows:
            logger.warning("No valid transactions found for batch.")
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        df = pd.DataFrame(feature_rows, index=found_ids)
        df = _apply_final_imputation(df)
        df = _enforce_column_order(df)

        logger.info("Batch transform complete: %d / %d rows produced.",
                    len(df), len(transaction_ids))
        return df

    # ------------------------------------------------------------------
    # Utility — compute features from raw dicts (no DB, for unit tests / API)
    # ------------------------------------------------------------------

    def transform_from_payload(
        self,
        txn: dict[str, Any],
        user: dict[str, Any],
        merchant: dict[str, Any],
        user_history: pd.DataFrame | None = None,
        merchant_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute features entirely from caller-supplied dicts and DataFrames.
        No database access — useful for:
          - Unit testing the pipeline without a live DB.
          - FastAPI endpoint that has already loaded dim data from cache.
          - Shadow scoring with pre-loaded feature windows.

        Args:
            txn:              Transaction dict (amount_usd, is_international, etc.)
            user:             User dict (account_age_days, risk_tier, etc.)
            merchant:         Merchant dict (risk_level, historical_fraud_rate, etc.)
            user_history:     Optional DataFrame of past user transactions.
            merchant_history: Optional DataFrame of past merchant transactions.

        Returns:
            pd.DataFrame with shape (1, len(FEATURE_COLUMNS)).
        """
        user_history     = user_history     if user_history     is not None else pd.DataFrame()
        merchant_history = merchant_history if merchant_history is not None else pd.DataFrame()

        # Build a unified row dict to match the shape _featurise_single_row expects
        row_dict: dict[str, Any] = {
            "amount_usd": txn.get("amount_usd", 0.0),
            "amount": txn.get("amount", txn.get("amount_usd", 0.0)),
            "is_international": txn.get("is_international", False),
            "channel": txn.get("channel", "web"),
            "payment_method": txn.get("payment_method", "credit_card"),
            "txn_timestamp": txn.get("txn_timestamp", datetime.now(timezone.utc)),
            "user_account_age_days": user.get("account_age_days", 365),
            "user_risk_tier": user.get("risk_tier", "medium"),
            "user_credit_score": user.get("credit_score"),
            "user_kyc_verified": user.get("kyc_verified", False),
            "merchant_risk_level": merchant.get("risk_level", "medium"),
            "merchant_historical_fraud_rate": merchant.get("historical_fraud_rate", 0.0),
            "merchant_is_high_risk_category": merchant.get("is_high_risk_category", False),
            "merchant_is_online_only": merchant.get("is_online_only", False),
            "merchant_avg_transaction_amount": merchant.get("avg_transaction_amount", 0.0),
        }
        row = pd.Series(row_dict)

        features = _featurise_single_row(row, user_history, merchant_history)
        df = pd.DataFrame([features])
        df = _apply_final_imputation(df)
        df = _enforce_column_order(df)
        return df
