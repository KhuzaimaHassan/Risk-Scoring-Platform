"""
src/features/feature_extraction.py
-------------------------------------
Raw feature extraction from a single transaction record and its associated
dimension tables (dim_user, dim_merchant).

Responsibility:
    Extract the immediate, non-aggregated attributes of a transaction that
    are known AT THE MOMENT the transaction is created. No historical lookback
    happens here — that is the job of feature_aggregation.py.

This layer is called by feature_pipeline.py for both training and inference.

Design rules:
    - Pure functions — no database access. All DB loading happens upstream
      in feature_pipeline.py, which passes pre-loaded dictionaries here.
    - No side effects — functions are idempotent.
    - Handles missing / nullable fields gracefully via defaults.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ordered list of all raw (non-aggregated) feature names produced here.
#: feature_pipeline.py uses this list to guarantee column ordering.
RAW_FEATURE_NAMES: list[str] = [
    # Transaction-level
    "amount_usd",
    "is_international",
    "is_online_merchant",
    # Temporal
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",          # 22:00–06:00 local hour
    # Channel / method encoding (ordinal, not OHE — tree models handle this)
    "channel_code",
    "payment_method_code",
    # User dimension
    "user_account_age_days",
    "user_risk_tier_code",
    "user_credit_score",
    "user_kyc_verified",
    # Merchant dimension
    "merchant_risk_level_code",
    "merchant_historical_fraud_rate",
    "merchant_is_high_risk_category",
    "merchant_avg_transaction_amount",
]

#: Ordinal mappings for categorical fields.
#: Using integer codes keeps feature vectors compact and avoids OHE explosion.
_CHANNEL_CODES: dict[str, int] = {
    "web": 0, "mobile": 1, "pos": 2, "atm": 3, "api": 4,
}
_PAYMENT_METHOD_CODES: dict[str, int] = {
    "credit_card": 0, "debit_card": 1, "bank_transfer": 2,
    "crypto": 3, "wallet": 4, "bnpl": 5,
}
_RISK_TIER_CODES: dict[str, int] = {
    "low": 0, "medium": 1, "high": 2, "blocked": 3,
}
_RISK_LEVEL_CODES: dict[str, int] = {
    "low": 0, "medium": 1, "high": 2,
}


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_temporal_features(txn_timestamp: datetime) -> dict[str, Any]:
    """
    Derive time-based features from the transaction timestamp.

    Args:
        txn_timestamp: Timezone-aware UTC datetime of the transaction.

    Returns:
        Dict with keys: hour_of_day, day_of_week, is_weekend, is_night.

    Note:
        All temporal features are UTC-based. In production you would use the
        user's local timezone — extend this function with a `tz_offset` param.
    """
    if txn_timestamp.tzinfo is None:
        txn_timestamp = txn_timestamp.replace(tzinfo=timezone.utc)

    hour = txn_timestamp.hour
    dow = txn_timestamp.weekday()  # 0 = Monday … 6 = Sunday

    return {
        "hour_of_day": hour,
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "is_night": int(hour >= 22 or hour < 6),
    }


def extract_transaction_features(txn: dict[str, Any]) -> dict[str, Any]:
    """
    Extract base features from the core transaction record.

    Args:
        txn: Dict representing one row from fact_transaction. Expected keys:
            amount_usd, is_international, channel, payment_method, txn_timestamp.

    Returns:
        Dict of raw transaction features.
    """
    timestamp: datetime = txn["txn_timestamp"]
    temporal = extract_temporal_features(timestamp)

    return {
        "amount_usd": float(txn.get("amount_usd", 0.0)),
        "is_international": int(bool(txn.get("is_international", False))),
        "channel_code": _CHANNEL_CODES.get(str(txn.get("channel", "web")), 0),
        "payment_method_code": _PAYMENT_METHOD_CODES.get(
            str(txn.get("payment_method", "credit_card")), 0
        ),
        **temporal,
    }


def extract_user_features(user: dict[str, Any]) -> dict[str, Any]:
    """
    Extract static (dimension-snapshot) features from a dim_user record.

    Args:
        user: Dict representing one row from dim_user. Expected keys:
            account_age_days, risk_tier, credit_score, kyc_verified.

    Returns:
        Dict of user-derived features.

    Note:
        credit_score is nullable — filled with the population median (670)
        when missing, which is a conservative imputation for tree models.
    """
    credit_score_raw = user.get("credit_score")
    credit_score: float = (
        float(credit_score_raw) if credit_score_raw is not None and credit_score_raw == credit_score_raw
        else 670.0
    )

    acct_age_raw = user.get("account_age_days", 0)
    acct_age = int(acct_age_raw) if acct_age_raw == acct_age_raw else 0  # NaN check

    return {
        "user_account_age_days": acct_age,
        "user_risk_tier_code": _RISK_TIER_CODES.get(
            str(user.get("risk_tier", "medium")), 1
        ),
        "user_credit_score": credit_score,
        "user_kyc_verified": int(bool(user.get("kyc_verified", False))),
    }



def extract_merchant_features(merchant: dict[str, Any]) -> dict[str, Any]:
    """
    Extract static features from a dim_merchant record.

    Args:
        merchant: Dict representing one row from dim_merchant. Expected keys:
            risk_level, historical_fraud_rate, is_high_risk_category,
            is_online_only, avg_transaction_amount.

    Returns:
        Dict of merchant-derived features.
    """
    return {
        "is_online_merchant": int(bool(merchant.get("is_online_only", False))),
        "merchant_risk_level_code": _RISK_LEVEL_CODES.get(
            str(merchant.get("risk_level", "medium")), 1
        ),
        "merchant_historical_fraud_rate": float(
            merchant.get("historical_fraud_rate", 0.0)
        ),
        "merchant_is_high_risk_category": int(
            bool(merchant.get("is_high_risk_category", False))
        ),
        "merchant_avg_transaction_amount": float(
            merchant.get("avg_transaction_amount", 0.0)
        ),
    }


def extract_raw_features(
    txn: dict[str, Any],
    user: dict[str, Any],
    merchant: dict[str, Any],
) -> dict[str, Any]:
    """
    Compose all raw features for a single transaction into one flat dict.

    This is the primary public function for extracting non-aggregated features.
    Call this before calling feature_aggregation functions.

    Args:
        txn:       Row dict from fact_transaction.
        user:      Row dict from dim_user (matched by user_id).
        merchant:  Row dict from dim_merchant (matched by merchant_id).

    Returns:
        Flat dict containing all raw features from RAW_FEATURE_NAMES.

    Example:
        raw = extract_raw_features(txn_dict, user_dict, merchant_dict)
        # {'amount_usd': 253.40, 'is_international': 0, ...}
    """
    features: dict[str, Any] = {}
    features.update(extract_transaction_features(txn))
    features.update(extract_user_features(user))
    features.update(extract_merchant_features(merchant))
    return features


def raw_features_to_series(
    txn: dict[str, Any],
    user: dict[str, Any],
    merchant: dict[str, Any],
) -> pd.Series:
    """
    Convenience wrapper: returns raw features as a pandas Series
    with a consistent index order (RAW_FEATURE_NAMES).

    Useful for unit testing individual transactions.
    """
    raw = extract_raw_features(txn, user, merchant)
    return pd.Series({k: raw.get(k, np.nan) for k in RAW_FEATURE_NAMES})
