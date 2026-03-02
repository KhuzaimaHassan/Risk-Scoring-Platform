"""
src/monitoring/drift_detection.py
-----------------------------------
Feature drift detection: compares the training-time feature distribution
against recent live data pulled from prediction_log.feature_vector.

Statistical tests used
----------------------
- KS-test   : Kolmogorov-Smirnov for continuous / numeric features.
- Chi-Square : For ordinal encoded categoricals (channel_code, etc.).

A feature is flagged as DRIFTED when  p-value < DRIFT_THRESHOLD (default 0.05).

Output
------
Saves a JSON drift report to reports/drift_reports/drift_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Bootstrap path so script can be run from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.feature_pipeline import FEATURE_COLUMNS  # noqa: E402 – path fixed above

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DRIFT_THRESHOLD: float = 0.05          # p-value below this = drift detected
LOOKBACK_DAYS:   int   = 7             # how many days of live predictions to use
MIN_LIVE_SAMPLES: int  = 30            # skip test if fewer live rows

# These feature names are ordinal-encoded categoricals (small integer domain).
# Chi-square is more appropriate than KS for them.
CATEGORICAL_FEATURES: set[str] = {
    "is_international",
    "is_online_merchant",
    "is_weekend",
    "is_night",
    "channel_code",
    "payment_method_code",
    "user_risk_tier_code",
    "user_kyc_verified",
    "merchant_risk_level_code",
    "merchant_is_high_risk_category",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_db_engine():
    """Create a sync SQLAlchemy engine from DATABASE_URL env var."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("SYNC_DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in environment / .env")
    # asyncpg URL → psycopg2 URL for sync engine
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )
    return create_engine(db_url, pool_pre_ping=True)


def _load_training_baseline(metadata_path: Path) -> pd.DataFrame | None:
    """
    Load feature statistics from the training metadata JSON.
    Returns a DataFrame with columns: [feature, mean, std, min, max, q25, q75]
    or None if not available.
    """
    if not metadata_path.exists():
        logger.warning("Metadata file not found: %s", metadata_path)
        return None
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    stats = meta.get("feature_stats")
    if not stats:
        logger.warning("No 'feature_stats' key in metadata — run train with save_feature_stats=True")
        return None
    return pd.DataFrame(stats)


def _load_live_features(engine, lookback_days: int) -> pd.DataFrame:
    """
    Pull feature_vector JSON from prediction_log for the last `lookback_days`.
    Returns a DataFrame where each column is a feature.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    sql = text("""
        SELECT feature_vector
        FROM   prediction_log
        WHERE  created_at >= :cutoff
          AND  feature_vector IS NOT NULL
        ORDER  BY created_at DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()

    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for (fv,) in rows:
        if isinstance(fv, dict):
            records.append(fv)
        elif isinstance(fv, str):
            try:
                records.append(json.loads(fv))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(records) if records else pd.DataFrame()


def _load_training_raw_features(engine, limit: int = 5000) -> pd.DataFrame:
    """
    Load a sample of training-era features directly from fact_transaction
    (joined with dim_user / dim_merchant) to build the baseline distribution.
    Used when metadata doesn't contain pre-computed feature_stats.
    """
    sql = text("""
        SELECT
            ft.amount_usd,
            ft.is_international,
            ft.channel,
            ft.payment_method,
            EXTRACT(HOUR FROM ft.txn_timestamp) AS hour_of_day,
            EXTRACT(DOW  FROM ft.txn_timestamp) AS day_of_week,
            du.account_age_days  AS user_account_age_days,
            du.risk_tier         AS user_risk_tier,
            du.credit_score      AS user_credit_score,
            du.kyc_verified      AS user_kyc_verified,
            dm.risk_level        AS merchant_risk_level,
            dm.historical_fraud_rate,
            dm.is_high_risk_category,
            dm.avg_transaction_amount
        FROM   fact_transaction ft
        JOIN   dim_user     du ON du.user_id     = ft.user_id
        JOIN   dim_merchant dm ON dm.merchant_id = ft.merchant_id
        WHERE  ft.fraud_label IS NOT NULL
        ORDER  BY RANDOM()
        LIMIT  :lim
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"lim": limit}).fetchall()
        cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'fact_transaction' ORDER BY ordinal_position
        """)).fetchall()

    if not rows:
        return pd.DataFrame()

    col_names = [
        "amount_usd", "is_international", "channel", "payment_method",
        "hour_of_day", "day_of_week", "user_account_age_days", "user_risk_tier",
        "user_credit_score", "user_kyc_verified", "merchant_risk_level",
        "historical_fraud_rate", "is_high_risk_category", "avg_transaction_amount",
    ]
    df = pd.DataFrame(rows, columns=col_names)

    # Encode categoricals the same way as feature_extraction.py
    channel_map = {"web": 0, "mobile": 1, "pos": 2, "atm": 3, "api": 4}
    pm_map      = {"credit_card": 0, "debit_card": 1, "bank_transfer": 2,
                   "crypto": 3, "wallet": 4, "bnpl": 5}
    tier_map    = {"low": 0, "medium": 1, "high": 2, "blocked": 3}
    risk_map    = {"low": 0, "medium": 1, "high": 2}

    df["channel_code"]          = df["channel"].map(channel_map).fillna(0).astype(int)
    df["payment_method_code"]   = df["payment_method"].map(pm_map).fillna(0).astype(int)
    df["user_risk_tier_code"]   = df["user_risk_tier"].map(tier_map).fillna(1).astype(int)
    df["merchant_risk_level_code"] = df["merchant_risk_level"].map(risk_map).fillna(1).astype(int)
    df["is_weekend"]            = df["day_of_week"].apply(lambda d: int(d >= 5))
    df["is_night"]              = df["hour_of_day"].apply(lambda h: int(h >= 22 or h < 6))
    df["user_kyc_verified"]     = df["user_kyc_verified"].astype(int)
    df["is_international"]      = df["is_international"].astype(int)
    df["is_online_merchant"]    = df["is_high_risk_category"].astype(int)
    df = df.rename(columns={
        "historical_fraud_rate": "merchant_historical_fraud_rate",
        "avg_transaction_amount": "merchant_avg_transaction_amount",
        "is_high_risk_category": "merchant_is_high_risk_category",
    })
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _run_ks_test(train_vals: np.ndarray, live_vals: np.ndarray) -> dict[str, float]:
    stat, pval = ks_2samp(train_vals, live_vals)
    return {"statistic": round(float(stat), 6), "p_value": round(float(pval), 6)}


def _run_chi2_test(train_vals: np.ndarray, live_vals: np.ndarray) -> dict[str, float]:
    """Chi-square on value_counts contingency table."""
    all_vals = sorted(set(train_vals.tolist()) | set(live_vals.tolist()))
    if len(all_vals) < 2:
        return {"statistic": 0.0, "p_value": 1.0}
    tc = np.array([np.sum(train_vals == v) for v in all_vals], dtype=float)
    lc = np.array([np.sum(live_vals  == v) for v in all_vals], dtype=float)
    # Avoid zero-expected
    tc = np.where(tc == 0, 0.5, tc)
    lc = np.where(lc == 0, 0.5, lc)
    table = np.array([tc, lc])
    try:
        stat, pval, _, _ = chi2_contingency(table)
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0}
    return {"statistic": round(float(stat), 6), "p_value": round(float(pval), 6)}


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def run_drift_detection(
    model_dir:   Path = ROOT / "models",
    report_dir:  Path = ROOT / "reports" / "drift_reports",
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Run drift detection comparing the training baseline to the last
    `lookback_days` of live predictions from prediction_log.

    Returns the full drift report dict (also saved to disk).
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    engine = _get_db_engine()

    # ── 1. Load training baseline ──────────────────────────────────────
    logger.info("Loading training baseline distribution …")
    train_df = _load_training_raw_features(engine, limit=5000)
    if train_df.empty:
        raise RuntimeError("Could not load training baseline from fact_transaction")
    logger.info("  Training baseline: %d rows", len(train_df))

    # ── 2. Load live features ─────────────────────────────────────────
    logger.info("Loading live features from prediction_log (last %d days) …", lookback_days)
    live_df = _load_live_features(engine, lookback_days)
    logger.info("  Live samples: %d", len(live_df))

    if len(live_df) < MIN_LIVE_SAMPLES:
        logger.warning(
            "Only %d live samples — fewer than minimum %d. "
            "Results may not be statistically meaningful.",
            len(live_df), MIN_LIVE_SAMPLES,
        )

    # ── 3. Compute drift per feature ──────────────────────────────────
    feature_results: list[dict[str, Any]] = []
    drifted_features: list[str] = []

    shared_cols = [c for c in train_df.columns if c in (live_df.columns if not live_df.empty else [])]
    if live_df.empty:
        shared_cols = []  # no live features to test

    for feat in FEATURE_COLUMNS:
        # Match feature name to available columns (raw names may differ slightly)
        train_col = feat if feat in train_df.columns else None
        live_col  = feat if (not live_df.empty and feat in live_df.columns) else None

        if train_col is None or live_col is None:
            feature_results.append({
                "feature":    feat,
                "test":       "skipped",
                "statistic":  None,
                "p_value":    None,
                "drifted":    False,
                "reason":     "column not available in baseline or live data",
            })
            continue

        train_vals = pd.to_numeric(train_df[train_col], errors="coerce").dropna().to_numpy()
        live_vals  = pd.to_numeric(live_df[live_col],  errors="coerce").dropna().to_numpy()

        if len(live_vals) < 2 or len(train_vals) < 2:
            feature_results.append({
                "feature": feat, "test": "skipped",
                "statistic": None, "p_value": None, "drifted": False,
                "reason": "insufficient data",
            })
            continue

        is_categorical = feat in CATEGORICAL_FEATURES
        if is_categorical:
            test_name = "chi2"
            result    = _run_chi2_test(train_vals.astype(int), live_vals.astype(int))
        else:
            test_name = "ks"
            result    = _run_ks_test(train_vals, live_vals)

        drifted = result["p_value"] < DRIFT_THRESHOLD
        if drifted:
            drifted_features.append(feat)

        feature_results.append({
            "feature":   feat,
            "test":      test_name,
            "statistic": result["statistic"],
            "p_value":   result["p_value"],
            "drifted":   drifted,
            "reason":    "p_value < 0.05" if drifted else "no drift detected",
            "train_mean": float(np.nanmean(train_vals)) if len(train_vals) else None,
            "live_mean":  float(np.nanmean(live_vals))  if len(live_vals)  else None,
        })

    # ── 4. Summary ───────────────────────────────────────────────────
    total_tested = sum(1 for r in feature_results if r["test"] not in ("skipped",))
    drift_score  = len(drifted_features) / max(total_tested, 1)

    report: dict[str, Any] = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "lookback_days":     lookback_days,
        "drift_threshold":   DRIFT_THRESHOLD,
        "train_samples":     len(train_df),
        "live_samples":      len(live_df),
        "features_tested":   total_tested,
        "features_drifted":  len(drifted_features),
        "drift_score":       round(drift_score, 4),   # 0 = no drift, 1 = all features drifted
        "alert_level":       _drift_alert_level(drift_score),
        "drifted_features":  drifted_features,
        "feature_details":   feature_results,
    }

    # ── 5. Save ──────────────────────────────────────────────────────
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = report_dir / f"drift_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Drift report saved: %s", out)

    # Also write a latest symlink-equivalent (plain copy)
    latest = report_dir / "drift_latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _log_summary(report)
    return report


def _drift_alert_level(score: float) -> str:
    if score < 0.10:
        return "green"   # < 10% features drifted
    if score < 0.30:
        return "yellow"  # 10-30% drifted — monitor closely
    return "red"         # > 30% drifted — retrain recommended


def _log_summary(report: dict[str, Any]) -> None:
    logger.info("=" * 55)
    logger.info("  DRIFT DETECTION SUMMARY")
    logger.info("=" * 55)
    logger.info("  Live samples    : %d (last %d days)",
                report["live_samples"], report["lookback_days"])
    logger.info("  Features tested : %d", report["features_tested"])
    logger.info("  Features drifted: %d", report["features_drifted"])
    logger.info("  Drift score     : %.2f", report["drift_score"])
    logger.info("  Alert level     : %s", report["alert_level"].upper())
    if report["drifted_features"]:
        logger.info("  Drifted         : %s", ", ".join(report["drifted_features"]))
    logger.info("=" * 55)
