"""
tests/unit/test_features.py
-----------------------------
Unit tests for the feature engineering layer.

Tests are structured to verify:
1. Raw feature extraction produces correct values and column types.
2. Rolling window aggregations respect the data-leakage boundary.
3. Zero-history (new entity) cases return safe sentinel values.
4. FeaturePipeline.transform_from_payload() produces the right shape.
5. Column ordering is always consistent with FEATURE_COLUMNS.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    FeaturePipeline,
    compute_all_aggregated_features,
    extract_raw_features,
    extract_temporal_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_timestamp() -> datetime:
    """A fixed UTC timestamp used across tests for reproducibility."""
    return datetime(2025, 12, 15, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_txn(base_timestamp) -> dict:
    return {
        "amount_usd": 250.00,
        "is_international": False,
        "channel": "web",
        "payment_method": "credit_card",
        "txn_timestamp": base_timestamp,
    }


@pytest.fixture
def sample_user() -> dict:
    return {
        "account_age_days": 730,
        "risk_tier": "medium",
        "credit_score": 720.0,
        "kyc_verified": True,
    }


@pytest.fixture
def sample_merchant() -> dict:
    return {
        "risk_level": "low",
        "historical_fraud_rate": 0.01,
        "is_high_risk_category": False,
        "is_online_only": True,
        "avg_transaction_amount": 150.0,
    }


@pytest.fixture
def user_history_df(base_timestamp) -> pd.DataFrame:
    """
    20 past transactions for a user, spread over the 7 days before base_timestamp.
    - 3 within the last 1 hour
    - 8 within the last 24 hours
    - 20 within the last 7 days
    """
    now = base_timestamp
    rows = []
    # 3 transactions in last 1 hour
    for i in range(3):
        rows.append({
            "txn_timestamp": now - timedelta(minutes=10 + i * 10),
            "amount_usd": 100.0 + i * 20,
            "fraud_label": 0,
        })
    # 5 more in last 24 hours
    for i in range(5):
        rows.append({
            "txn_timestamp": now - timedelta(hours=2 + i * 2),
            "amount_usd": 50.0 + i * 15,
            "fraud_label": 0,
        })
    # 12 more in last 7 days
    for i in range(12):
        rows.append({
            "txn_timestamp": now - timedelta(days=1 + i * 0.4),
            "amount_usd": 80.0 + i * 10,
            "fraud_label": 0,
        })
    df = pd.DataFrame(rows)
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
    return df


@pytest.fixture
def merchant_history_df(base_timestamp) -> pd.DataFrame:
    """15 past transactions at a merchant, 1 marked as fraud."""
    now = base_timestamp
    rows = []
    for i in range(14):
        rows.append({
            "txn_timestamp": now - timedelta(days=i * 0.4),
            "amount_usd": 120.0 + i * 5,
            "fraud_label": 0,
        })
    # 1 fraud row
    rows.append({
        "txn_timestamp": now - timedelta(days=2),
        "amount_usd": 900.0,
        "fraud_label": 1,
    })
    df = pd.DataFrame(rows)
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# 1. Temporal feature tests
# ---------------------------------------------------------------------------

class TestTemporalFeatures:

    def test_hour_extraction(self):
        ts = datetime(2025, 6, 10, 9, 45, tzinfo=timezone.utc)
        feats = extract_temporal_features(ts)
        assert feats["hour_of_day"] == 9

    def test_weekend_flag(self):
        # 2025-06-14 is a Saturday
        saturday = datetime(2025, 6, 14, 12, 0, tzinfo=timezone.utc)
        feats = extract_temporal_features(saturday)
        assert feats["is_weekend"] == 1

    def test_weekday_flag(self):
        # 2025-06-10 is a Tuesday
        tuesday = datetime(2025, 6, 10, 12, 0, tzinfo=timezone.utc)
        feats = extract_temporal_features(tuesday)
        assert feats["is_weekend"] == 0

    def test_night_flag_at_23h(self):
        ts = datetime(2025, 6, 10, 23, 15, tzinfo=timezone.utc)
        feats = extract_temporal_features(ts)
        assert feats["is_night"] == 1

    def test_not_night_at_noon(self):
        ts = datetime(2025, 6, 10, 12, 0, tzinfo=timezone.utc)
        feats = extract_temporal_features(ts)
        assert feats["is_night"] == 0

    def test_naive_datetime_handled(self):
        """Naive datetimes should be accepted without raising."""
        ts = datetime(2025, 6, 10, 14, 0)  # no tzinfo
        feats = extract_temporal_features(ts)
        assert "hour_of_day" in feats


# ---------------------------------------------------------------------------
# 2. Raw feature extraction tests
# ---------------------------------------------------------------------------

class TestRawFeatureExtraction:

    def test_returns_all_raw_feature_keys(self, sample_txn, sample_user, sample_merchant):
        from src.features import RAW_FEATURE_NAMES
        raw = extract_raw_features(sample_txn, sample_user, sample_merchant)
        for key in RAW_FEATURE_NAMES:
            assert key in raw, f"Missing raw feature: {key}"

    def test_amount_usd_passthrough(self, sample_txn, sample_user, sample_merchant):
        raw = extract_raw_features(sample_txn, sample_user, sample_merchant)
        assert raw["amount_usd"] == pytest.approx(250.00)

    def test_is_international_flag(self, sample_txn, sample_user, sample_merchant):
        raw = extract_raw_features(sample_txn, sample_user, sample_merchant)
        assert raw["is_international"] == 0

    def test_credit_score_null_imputation(self, sample_txn, sample_merchant):
        user_no_score = {"account_age_days": 365, "risk_tier": "medium",
                         "credit_score": None, "kyc_verified": True}
        raw = extract_raw_features(sample_txn, user_no_score, sample_merchant)
        # Should impute to 670 (population median)
        assert raw["user_credit_score"] == pytest.approx(670.0)

    def test_channel_code_encoding(self, sample_txn, sample_user, sample_merchant):
        raw = extract_raw_features(sample_txn, sample_user, sample_merchant)
        assert raw["channel_code"] == 0   # "web" → 0

    def test_unknown_channel_defaults_to_zero(self, sample_txn, sample_user, sample_merchant):
        txn = {**sample_txn, "channel": "unknown_channel"}
        raw = extract_raw_features(txn, sample_user, sample_merchant)
        assert raw["channel_code"] == 0


# ---------------------------------------------------------------------------
# 3. Rolling window aggregation tests
# ---------------------------------------------------------------------------

class TestRollingAggregation:

    def test_user_count_1h(self, user_history_df, base_timestamp):
        feats = compute_all_aggregated_features(
            user_history=user_history_df,
            merchant_history=pd.DataFrame(),
            current_ts=base_timestamp,
            current_amount_usd=200.0,
        )
        assert feats["user_txn_count_1h"] == 3

    def test_user_count_24h(self, user_history_df, base_timestamp):
        feats = compute_all_aggregated_features(
            user_history=user_history_df,
            merchant_history=pd.DataFrame(),
            current_ts=base_timestamp,
            current_amount_usd=200.0,
        )
        # 3 in 1h + 5 in 2–24h = 8
        assert feats["user_txn_count_24h"] == 8

    def test_user_count_7d(self, user_history_df, base_timestamp):
        feats = compute_all_aggregated_features(
            user_history=user_history_df,
            merchant_history=pd.DataFrame(),
            current_ts=base_timestamp,
            current_amount_usd=200.0,
        )
        assert feats["user_txn_count_7d"] == 20

    def test_no_data_leakage_when_future_rows_present(self, base_timestamp):
        """Future rows (after current_ts) must NOT be counted."""
        future_rows = pd.DataFrame([{
            "txn_timestamp": pd.Timestamp(base_timestamp + timedelta(hours=1), tz="UTC"),
            "amount_usd": 9999.0,
            "fraud_label": 0,
        }])
        feats = compute_all_aggregated_features(
            user_history=future_rows,
            merchant_history=pd.DataFrame(),
            current_ts=base_timestamp,
            current_amount_usd=100.0,
        )
        assert feats["user_txn_count_7d"] == 0, "Future row must not be counted"

    def test_zero_history_returns_safe_defaults(self, base_timestamp):
        """New user with no past transactions — all counts should be 0."""
        feats = compute_all_aggregated_features(
            user_history=pd.DataFrame(),
            merchant_history=pd.DataFrame(),
            current_ts=base_timestamp,
            current_amount_usd=100.0,
        )
        assert feats["user_txn_count_1h"] == 0
        assert feats["user_txn_count_24h"] == 0
        assert feats["user_txn_count_7d"] == 0
        assert feats["user_avg_amount_7d"] == pytest.approx(0.0)
        assert feats["time_since_last_txn_minutes"] == pytest.approx(43200.0)

    def test_amount_vs_user_avg_ratio_high(self, user_history_df, base_timestamp):
        """A very high amount should produce a ratio >> 1.0."""
        feats = compute_all_aggregated_features(
            user_history=user_history_df,
            merchant_history=pd.DataFrame(),
            current_ts=base_timestamp,
            current_amount_usd=10_000.0,  # Very high
        )
        assert feats["amount_vs_user_avg_ratio"] > 1.0

    def test_merchant_fraud_rate_7d(self, merchant_history_df, base_timestamp):
        """Merchant history has 1 fraud out of 15 transactions in 7d window."""
        feats = compute_all_aggregated_features(
            user_history=pd.DataFrame(),
            merchant_history=merchant_history_df,
            current_ts=base_timestamp,
            current_amount_usd=150.0,
        )
        # Should be 1/15 ≈ 0.0667
        assert 0.05 < feats["merchant_fraud_rate_7d"] < 0.10

    def test_time_since_last_txn_accuracy(self, base_timestamp):
        """Verify time_since_last_txn_minutes is computed correctly."""
        last_txn_ts = base_timestamp - timedelta(minutes=90)
        history = pd.DataFrame([{
            "txn_timestamp": pd.Timestamp(last_txn_ts, tz="UTC"),
            "amount_usd": 100.0,
            "fraud_label": 0,
        }])
        feats = compute_all_aggregated_features(
            user_history=history,
            merchant_history=pd.DataFrame(),
            current_ts=base_timestamp,
            current_amount_usd=100.0,
        )
        assert feats["time_since_last_txn_minutes"] == pytest.approx(90.0, abs=0.1)


# ---------------------------------------------------------------------------
# 4. FeaturePipeline.transform_from_payload tests
# ---------------------------------------------------------------------------

class TestFeaturePipelinePayload:

    @pytest.fixture
    def pipeline(self) -> FeaturePipeline:
        return FeaturePipeline(lookback_days=7)

    def test_output_shape(self, pipeline, sample_txn, sample_user, sample_merchant):
        df = pipeline.transform_from_payload(sample_txn, sample_user, sample_merchant)
        assert df.shape == (1, len(FEATURE_COLUMNS))

    def test_column_order_matches_contract(self, pipeline, sample_txn, sample_user, sample_merchant):
        df = pipeline.transform_from_payload(sample_txn, sample_user, sample_merchant)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_no_nan_in_output(self, pipeline, sample_txn, sample_user, sample_merchant):
        df = pipeline.transform_from_payload(sample_txn, sample_user, sample_merchant)
        assert not df.isna().any().any(), "Output must contain no NaN values"

    def test_with_history_counts_propagate(
        self, pipeline, sample_txn, sample_user, sample_merchant,
        user_history_df, merchant_history_df, base_timestamp
    ):
        df = pipeline.transform_from_payload(
            sample_txn, sample_user, sample_merchant,
            user_history=user_history_df,
            merchant_history=merchant_history_df,
        )
        assert df["user_txn_count_24h"].iloc[0] == 8

    def test_invalid_lookback_raises(self):
        with pytest.raises(ValueError, match="lookback_days"):
            FeaturePipeline(lookback_days=5)

    def test_all_features_numeric(self, pipeline, sample_txn, sample_user, sample_merchant):
        df = pipeline.transform_from_payload(sample_txn, sample_user, sample_merchant)
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        assert non_numeric == [], f"Non-numeric feature columns: {non_numeric}"
