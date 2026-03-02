"""
src/features/feature_aggregation.py
--------------------------------------
Rolling-window and behavioural aggregate feature computation.

Responsibility:
    Given a DataFrame of HISTORICAL transactions for a user/merchant
    (i.e. all rows BEFORE the current transaction's timestamp), compute
    statistical aggregates that summarise past behaviour.

Data leakage contract:
    All functions receive `history_df` which must contain ONLY transactions
    with txn_timestamp < current_txn_timestamp. The caller (feature_pipeline.py)
    enforces this strict inequality — functions here do NOT re-filter.

Design rules:
    - Pure functions — accept DataFrames, return dicts.
    - No DB access — all data is passed in.
    - Safe for zero-history cases (new users / merchants).
    - Vectorised via pandas — no Python for-loops over rows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ordered list of all aggregated feature names produced in this module.
AGGREGATED_FEATURE_NAMES: list[str] = [
    # User rolling counts
    "user_txn_count_1h",
    "user_txn_count_24h",
    "user_txn_count_7d",
    # User rolling amounts
    "user_avg_amount_7d",
    "user_std_amount_7d",
    "user_max_amount_7d",
    "user_total_amount_24h",
    # User velocity ratio (1h count / 24h count — detects bursts)
    "user_velocity_ratio_1h_24h",
    # Behavioural
    "time_since_last_txn_minutes",
    "amount_vs_user_avg_ratio",
    "amount_vs_merchant_avg_ratio",
    # Merchant rolling
    "merchant_txn_count_7d",
    "merchant_fraud_rate_7d",       # Fraction of labelled-fraud txns in 7d window
    "merchant_avg_amount_7d",
]

#: Time window durations used throughout this module.
_WINDOW_1H  = timedelta(hours=1)
_WINDOW_24H = timedelta(hours=24)
_WINDOW_7D  = timedelta(days=7)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_window(
    history_df: pd.DataFrame,
    current_ts: datetime,
    window: timedelta,
    ts_col: str = "txn_timestamp",
) -> pd.DataFrame:
    """
    Return rows of `history_df` whose timestamp falls within
    [current_ts - window, current_ts) — exclusive of current_ts
    to prevent data leakage.

    Args:
        history_df:  DataFrame of historical (past-only) transactions.
        current_ts:  Timestamp of the transaction being featurised.
        window:      Lookback duration (timedelta).
        ts_col:      Name of the timestamp column in history_df.

    Returns:
        Filtered DataFrame (may be empty for new entities).
    """
    if history_df.empty:
        return history_df
    cutoff = current_ts - window
    mask = (history_df[ts_col] >= cutoff) & (history_df[ts_col] < current_ts)
    return history_df.loc[mask]


def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    """Return mean of series, or `default` if empty / all-NaN."""
    if series.empty:
        return default
    val = series.mean()
    return float(val) if not np.isnan(val) else default


def _safe_std(series: pd.Series, default: float = 0.0) -> float:
    """Return std dev of series, or `default` if fewer than 2 observations."""
    if len(series) < 2:
        return default
    val = series.std(ddof=1)
    return float(val) if not np.isnan(val) else default


def _safe_max(series: pd.Series, default: float = 0.0) -> float:
    """Return max of series, or `default` if empty."""
    if series.empty:
        return default
    return float(series.max())


# ---------------------------------------------------------------------------
# User-level rolling features
# ---------------------------------------------------------------------------

def compute_user_rolling_features(
    user_history: pd.DataFrame,
    current_ts: datetime,
    current_amount_usd: float,
) -> dict[str, Any]:
    """
    Compute rolling window features for the current user.

    Args:
        user_history:       DataFrame of past transactions for this user.
                            Required columns: txn_timestamp, amount_usd.
                            Must contain ONLY rows where txn_timestamp < current_ts.
        current_ts:         Timestamp of the transaction being scored.
        current_amount_usd: USD amount of the current transaction (used for ratio).

    Returns:
        Dict containing user rolling feature values from AGGREGATED_FEATURE_NAMES.
    """
    # Ensure tz-aware for window arithmetic
    if not user_history.empty:
        ts_col = user_history["txn_timestamp"]
        if hasattr(ts_col.dtype, "tz") and ts_col.dtype.tz is None:
            user_history = user_history.copy()
            user_history["txn_timestamp"] = ts_col.dt.tz_localize("UTC")

    if current_ts.tzinfo is None:
        current_ts = current_ts.replace(tzinfo=timezone.utc)

    w1h  = _filter_window(user_history, current_ts, _WINDOW_1H)
    w24h = _filter_window(user_history, current_ts, _WINDOW_24H)
    w7d  = _filter_window(user_history, current_ts, _WINDOW_7D)

    count_1h  = len(w1h)
    count_24h = len(w24h)
    count_7d  = len(w7d)

    avg_7d = _safe_mean(w7d["amount_usd"], default=0.0) if not w7d.empty else 0.0
    std_7d = _safe_std(w7d["amount_usd"], default=0.0)  if not w7d.empty else 0.0
    max_7d = _safe_max(w7d["amount_usd"], default=0.0)  if not w7d.empty else 0.0

    total_24h = float(w24h["amount_usd"].sum()) if not w24h.empty else 0.0

    # Velocity ratio: burst detection (e.g. 5 txns in 1h, 6 in 24h → ratio ≈ 0.83)
    velocity_ratio = (count_1h / count_24h) if count_24h > 0 else 0.0

    # Amount vs. 7d average ratio — key fraud signal
    amount_vs_avg = (current_amount_usd / avg_7d) if avg_7d > 0 else 1.0

    return {
        "user_txn_count_1h": count_1h,
        "user_txn_count_24h": count_24h,
        "user_txn_count_7d": count_7d,
        "user_avg_amount_7d": round(avg_7d, 4),
        "user_std_amount_7d": round(std_7d, 4),
        "user_max_amount_7d": round(max_7d, 4),
        "user_total_amount_24h": round(total_24h, 4),
        "user_velocity_ratio_1h_24h": round(velocity_ratio, 6),
        "amount_vs_user_avg_ratio": round(amount_vs_avg, 6),
    }


# ---------------------------------------------------------------------------
# Merchant-level rolling features
# ---------------------------------------------------------------------------

def compute_merchant_rolling_features(
    merchant_history: pd.DataFrame,
    current_ts: datetime,
    current_amount_usd: float,
    static_merchant_avg: float = 0.0,
) -> dict[str, Any]:
    """
    Compute rolling window features for the current merchant.

    Args:
        merchant_history:   DataFrame of past transactions at this merchant.
                            Required columns: txn_timestamp, amount_usd,
                            fraud_label (int, nullable).
        current_ts:         Timestamp of the transaction being scored.
        current_amount_usd: USD amount of the current transaction.
        static_merchant_avg: Lifetime avg amount from dim_merchant (fallback
                              when window history is insufficient).

    Returns:
        Dict containing merchant rolling features.
    """
    if current_ts.tzinfo is None:
        current_ts = current_ts.replace(tzinfo=timezone.utc)

    if not merchant_history.empty:
        ts_col = merchant_history["txn_timestamp"]
        if hasattr(ts_col.dtype, "tz") and ts_col.dtype.tz is None:
            merchant_history = merchant_history.copy()
            merchant_history["txn_timestamp"] = ts_col.dt.tz_localize("UTC")

    w7d = _filter_window(merchant_history, current_ts, _WINDOW_7D)

    count_7d = len(w7d)
    avg_7d_merchant = (
        _safe_mean(w7d["amount_usd"], default=static_merchant_avg)
        if not w7d.empty
        else static_merchant_avg
    )

    # Fraud rate within 7d window (only from labelled rows to avoid NULL bias)
    labelled_7d = w7d[w7d["fraud_label"].notna()] if not w7d.empty else pd.DataFrame()
    if not labelled_7d.empty:
        fraud_rate_7d = float(labelled_7d["fraud_label"].mean())
    else:
        fraud_rate_7d = 0.0

    # Amount deviation from merchant's 7d average
    fallback_avg = avg_7d_merchant if avg_7d_merchant > 0 else (static_merchant_avg or 1.0)
    amount_vs_merchant_avg = current_amount_usd / fallback_avg

    return {
        "merchant_txn_count_7d": count_7d,
        "merchant_fraud_rate_7d": round(fraud_rate_7d, 6),
        "merchant_avg_amount_7d": round(avg_7d_merchant, 4),
        "amount_vs_merchant_avg_ratio": round(amount_vs_merchant_avg, 6),
    }


# ---------------------------------------------------------------------------
# Behavioural features
# ---------------------------------------------------------------------------

def compute_behavioral_features(
    user_history: pd.DataFrame,
    current_ts: datetime,
) -> dict[str, Any]:
    """
    Compute cross-dimensional behavioural features.

    Currently includes:
        time_since_last_txn_minutes — minutes since this user's most recent
        past transaction. High values may indicate dormant accounts suddenly
        re-activated (account takeover signal).

    Args:
        user_history: DataFrame of past user transactions (txn_timestamp required).
        current_ts:   Timestamp of the current transaction.

    Returns:
        Dict of behavioural feature values.
    """
    if current_ts.tzinfo is None:
        current_ts = current_ts.replace(tzinfo=timezone.utc)

    if user_history.empty:
        # New user with no prior transactions — set sentinel value
        time_since_last = 43_200.0  # 30 days in minutes (capped sentinel)
    else:
        ts_series = user_history["txn_timestamp"]
        if hasattr(ts_series.dtype, "tz") and ts_series.dtype.tz is None:
            ts_series = ts_series.dt.tz_localize("UTC")

        past_ts = ts_series[ts_series < current_ts]
        if past_ts.empty:
            time_since_last = 43_200.0
        else:
            last_ts = past_ts.max()
            delta_minutes = (current_ts - last_ts).total_seconds() / 60.0
            # Cap at 30 days to avoid infinite-range outliers
            time_since_last = round(min(delta_minutes, 43_200.0), 2)

    return {
        "time_since_last_txn_minutes": time_since_last,
    }


# ---------------------------------------------------------------------------
# Combined aggregation entry point
# ---------------------------------------------------------------------------

def compute_all_aggregated_features(
    user_history: pd.DataFrame,
    merchant_history: pd.DataFrame,
    current_ts: datetime,
    current_amount_usd: float,
    static_merchant_avg: float = 0.0,
) -> dict[str, Any]:
    """
    Single entry point that computes all aggregated features.

    Called by feature_pipeline.py for both training and inference modes.

    Args:
        user_history:         Past transactions for the current user.
        merchant_history:     Past transactions at the current merchant.
        current_ts:           Timestamp of the current transaction.
        current_amount_usd:   USD amount of the current transaction.
        static_merchant_avg:  Lifetime avg from dim_merchant (fallback).

    Returns:
        Flat dict containing all keys from AGGREGATED_FEATURE_NAMES.
    """
    user_feats = compute_user_rolling_features(
        user_history, current_ts, current_amount_usd
    )
    merchant_feats = compute_merchant_rolling_features(
        merchant_history, current_ts, current_amount_usd, static_merchant_avg
    )
    behavioral_feats = compute_behavioral_features(user_history, current_ts)

    return {**user_feats, **merchant_feats, **behavioral_feats}
