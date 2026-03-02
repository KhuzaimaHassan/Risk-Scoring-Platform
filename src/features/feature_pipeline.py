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
FEATURE_COLUMNS: list[str] = RAW_FEATURE_NAMES + AGGREGATED_FEATURE_NAMES

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
            ft.fraud_label,
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
        WHERE  ft.status = 'completed'
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
            ft.fraud_label,
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
        SELECT txn_timestamp, amount_usd, fraud_label
        FROM   fact_transaction
        WHERE  user_id  = :user_id
          AND  txn_timestamp < :upper
          AND  txn_timestamp >= :lower
          AND  status = 'completed'
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
        SELECT txn_timestamp, amount_usd, fraud_label
        FROM   fact_transaction
        WHERE  merchant_id = :merchant_id
          AND  txn_timestamp < :upper
          AND  txn_timestamp >= :lower
          AND  status = 'completed'
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
    merchant_dict: dict[str, Any] = {
        "risk_level": txn_row["merchant_risk_level"],
        "historical_fraud_rate": txn_row["merchant_historical_fraud_rate"],
        "is_high_risk_category": txn_row["merchant_is_high_risk_category"],
        "is_online_only": txn_row["merchant_is_online_only"],
        "avg_transaction_amount": txn_row["merchant_avg_transaction_amount"],
    }

    # 1. Raw features (no historical lookback)
    raw_feats = extract_raw_features(txn_dict, user_dict, merchant_dict)

    # 2. Aggregated rolling features (strictly historical)
    agg_feats = compute_all_aggregated_features(
        user_history=user_history,
        merchant_history=merchant_history,
        current_ts=txn_row["txn_timestamp"],
        current_amount_usd=float(txn_row["amount_usd"]),
        static_merchant_avg=float(txn_row.get("merchant_avg_transaction_amount", 0.0)),
    )

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

    def __init__(self, lookback_days: int = 7) -> None:
        if lookback_days < 7:
            raise ValueError("lookback_days must be >= 7 to cover the 7-day rolling window.")
        self.lookback_days = lookback_days
        logger.info("FeaturePipeline initialised (lookback_days=%d).", lookback_days)

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------

    def build_training_dataset(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build the full feature matrix for model training.

        For each labelled transaction (fraud_label is not NULL), computes
        features using ONLY prior transactions (strict leakage prevention).

        The function processes transactions in chronological order so that
        each transaction's rolling window correctly excludes future rows.

        Args:
            since: Only include transactions on or after this datetime (UTC).
            until: Only include transactions before this datetime (UTC).

        Returns:
            Tuple of:
                X — pd.DataFrame shape (n_samples, len(FEATURE_COLUMNS))
                y — pd.Series[int] shape (n_samples,), fraud_label values

        Note:
            This method loads ALL relevant historical data into memory in one
            query, then computes rolling windows with in-memory pandas operations.
            For datasets > 200K rows, consider chunked processing.
        """
        logger.info("Building training dataset (since=%s, until=%s) …", since, until)

        with sync_db_context() as db:
            # Load all labelled completed transactions
            all_txns = _load_full_transaction_table(db, since=since, labelled_only=True)

        if all_txns.empty:
            logger.warning("No labelled transactions found. Returning empty dataset.")
            return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series([], dtype=int)

        if until:
            all_txns = all_txns[all_txns["txn_timestamp"] < pd.Timestamp(until, tz="UTC")]

        logger.info("Featurising %d labelled transactions (vectorised) …", len(all_txns))

        # Sort chronologically globally
        df = all_txns.sort_values("txn_timestamp").reset_index(drop=True)

        # ── Vectorized rolling helper ─────────────────────────────────────
        # Iterate explicitly over groups (pandas groupby.apply strips key col).
        # With 1,000 unique users and 100 merchants this is very fast.

        def _rolling_for_group(
            src: pd.DataFrame,
            group_col: str,
            val_col: str,
            windows: dict[str, str],   # feat_name -> window_string
            extra_cols: list[str] | None = None,
        ) -> pd.DataFrame:
            """
            For each group in `src[group_col]`, sort by txn_timestamp,
            then compute time-based rolling windows with shift(1) so the
            current row is never included in its own window.
            Returns a DataFrame with group_col, txn_timestamp, and all
            window feature columns.
            """
            pieces: list[pd.DataFrame] = []
            extra_cols = extra_cols or []
            for gid, grp in src.groupby(group_col, sort=False):
                g = grp.sort_values("txn_timestamp").copy()
                g = g.set_index("txn_timestamp")
                amt = g[val_col].shift(1)
                for feat_name, win in windows.items():
                    if "count" in feat_name:
                        g[feat_name] = amt.rolling(win, min_periods=0).count().fillna(0)
                    elif "avg" in feat_name or "mean" in feat_name:
                        g[feat_name] = amt.rolling(win, min_periods=1).mean().fillna(0)
                    elif "std" in feat_name:
                        g[feat_name] = amt.rolling(win, min_periods=2).std(ddof=1).fillna(0)
                    elif "max" in feat_name:
                        g[feat_name] = amt.rolling(win, min_periods=1).max().fillna(0)
                    elif "sum" in feat_name or "total" in feat_name:
                        g[feat_name] = amt.rolling(win, min_periods=0).sum().fillna(0)
                for ec in extra_cols:
                    if ec in grp.columns:
                        g[ec] = grp.set_index("txn_timestamp")[ec]
                g[group_col] = gid
                pieces.append(g.reset_index())  # txn_timestamp back to column
            return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

        # ── USER ROLLING ──────────────────────────────────────────────────
        udf = df[["user_id", "txn_timestamp", "amount_usd"]].copy()
        udf_rolled = _rolling_for_group(udf, "user_id", "amount_usd", {
            "user_txn_count_1h":    "3600s",
            "user_txn_count_24h":   "86400s",
            "user_txn_count_7d":    "604800s",
            "user_avg_amount_7d":   "604800s",
            "user_std_amount_7d":   "604800s",
            "user_max_amount_7d":   "604800s",
            "user_total_amount_24h":"86400s",
        })

        udf_rolled["user_velocity_ratio_1h_24h"] = (
            udf_rolled["user_txn_count_1h"] / udf_rolled["user_txn_count_24h"].clip(lower=1)
        )
        udf_rolled["amount_vs_user_avg_ratio"] = (
            udf_rolled["amount_usd"] / udf_rolled["user_avg_amount_7d"].replace(0, np.nan)
        ).fillna(1.0)

        udf_rolled = udf_rolled.sort_values(["user_id", "txn_timestamp"])
        udf_rolled["_prev_ts"] = udf_rolled.groupby("user_id")["txn_timestamp"].shift(1)
        udf_rolled["time_since_last_txn_minutes"] = (
            (udf_rolled["txn_timestamp"] - udf_rolled["_prev_ts"]).dt.total_seconds() / 60.0
        ).clip(upper=43_200.0).fillna(43_200.0)

        user_feat_cols = [
            "user_txn_count_1h", "user_txn_count_24h", "user_txn_count_7d",
            "user_avg_amount_7d", "user_std_amount_7d", "user_max_amount_7d",
            "user_total_amount_24h", "user_velocity_ratio_1h_24h",
            "amount_vs_user_avg_ratio", "time_since_last_txn_minutes",
        ]
        df = df.reset_index(drop=True)
        user_feats_aligned = df[["user_id", "txn_timestamp"]].merge(
            udf_rolled[["user_id", "txn_timestamp"] + user_feat_cols],
            on=["user_id", "txn_timestamp"], how="left",
        )[user_feat_cols]
        df = pd.concat([df, user_feats_aligned.reset_index(drop=True)], axis=1)

        # ── MERCHANT ROLLING ──────────────────────────────────────────────
        mdf = df[["merchant_id", "txn_timestamp", "amount_usd", "fraud_label"]].copy()
        # For merchant fraud rate we need a special handler (extra col)
        pieces_m: list[pd.DataFrame] = []
        for mid, grp in mdf.groupby("merchant_id", sort=False):
            g = grp.sort_values("txn_timestamp").copy().set_index("txn_timestamp")
            amt  = g["amount_usd"].shift(1)
            flab = g["fraud_label"].shift(1)
            g["merchant_txn_count_7d"]  = amt.rolling("604800s", min_periods=0).count().fillna(0)
            g["merchant_avg_amount_7d"] = amt.rolling("604800s", min_periods=1).mean().fillna(0)
            g["merchant_fraud_rate_7d"] = flab.rolling("604800s", min_periods=1).mean().fillna(0)
            g["merchant_id"] = mid
            pieces_m.append(g.reset_index())
        mdf_rolled = pd.concat(pieces_m, ignore_index=True) if pieces_m else pd.DataFrame()

        merchant_feat_cols = ["merchant_txn_count_7d", "merchant_avg_amount_7d", "merchant_fraud_rate_7d"]
        merchant_feats_aligned = df[["merchant_id", "txn_timestamp"]].merge(
            mdf_rolled[["merchant_id", "txn_timestamp"] + merchant_feat_cols],
            on=["merchant_id", "txn_timestamp"], how="left",
        )[merchant_feat_cols]
        df = pd.concat([df, merchant_feats_aligned.reset_index(drop=True)], axis=1)

        df["amount_vs_merchant_avg_ratio"] = (
            df["amount_usd"] / df["merchant_avg_amount_7d"].replace(0, np.nan)
        ).fillna(1.0)

        # ── RAW FEATURES (static per-row, fast) ───────────────────────────
        from src.features.feature_extraction import extract_raw_features

        raw_rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            raw_rows.append(extract_raw_features(
                txn={
                    "amount_usd":       float(row["amount_usd"]),
                    "is_international": bool(row["is_international"]),
                    "channel":          str(row["channel"]),
                    "payment_method":   str(row["payment_method"]),
                    "txn_timestamp":    row["txn_timestamp"],
                },
                user={
                    "account_age_days": row["user_account_age_days"],
                    "risk_tier":        str(row["user_risk_tier"]),
                    "credit_score":     row.get("user_credit_score"),
                    "kyc_verified":     bool(row["user_kyc_verified"]),
                },
                merchant={
                    "risk_level":             str(row["merchant_risk_level"]),
                    "historical_fraud_rate":  float(row["merchant_historical_fraud_rate"]),
                    "is_high_risk_category":  bool(row["merchant_is_high_risk_category"]),
                    "is_online_only":         bool(row["merchant_is_online_only"]),
                    "avg_transaction_amount": float(row["merchant_avg_transaction_amount"]),
                },
            ))

        raw_df = pd.DataFrame(raw_rows, index=df.index)
        agg_cols_present = [c for c in AGGREGATED_FEATURE_NAMES if c in df.columns]
        X = pd.concat([raw_df, df[agg_cols_present]], axis=1)

        # Drop any rows where fraud_label is still NaN after the labelled_only filter
        valid_mask = df["fraud_label"].notna()
        X = X.loc[valid_mask].reset_index(drop=True)
        y = pd.Series(
            df.loc[valid_mask, "fraud_label"].astype(int).values,
            dtype=int, name=TARGET_COLUMN,
        )


        X = _apply_final_imputation(X)
        X = _enforce_column_order(X)

        fraud_pct = y.mean() * 100
        logger.info(
            "Training dataset ready: %d samples, %d features, %.2f%% fraud.",
            len(X), len(FEATURE_COLUMNS), fraud_pct,
        )
        return X, y

    # ------------------------------------------------------------------
    # Inference mode — single transaction
    # ------------------------------------------------------------------



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
