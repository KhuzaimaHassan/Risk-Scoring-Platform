"""
src/monitoring/prediction_monitor.py
--------------------------------------
Tracks live prediction distribution metrics and compares them against
the training-time baseline.

Metrics tracked
---------------
- Mean fraud probability (score distribution)
- Fraud prediction rate (% of predictions flagged as fraud)
- Risk band distribution (low / medium / high / critical)
- Score percentiles (p50, p90, p95, p99)
- Mean inference latency

Comparison window
-----------------
Last 7 days of production predictions vs training baseline rate (2.62%).

Output
------
Saves a JSON snapshot to reports/monitoring_snapshots/snapshot_{ts}.json
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
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOOKBACK_DAYS:          int   = 7
TRAIN_FRAUD_RATE:       float = 0.0308   # 3.08% — from training dataset
FRAUD_RATE_ALERT_MULT:  float = 2.0      # alert if live fraud rate > 2x training
SCORE_MEAN_ALERT_DELTA: float = 0.10     # alert if live mean score drifts > 0.10


def _get_db_engine():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("SYNC_DATABASE_URL", "")
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )
    return create_engine(db_url, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _fetch_recent_predictions(engine, lookback_days: int) -> pd.DataFrame:
    """Pull fraud_score, risk_label, latency_ms from prediction_log."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    sql = text("""
        SELECT
            fraud_score,
            risk_label,
            decision_threshold,
            latency_ms,
            model_version,
            created_at
        FROM   prediction_log
        WHERE  created_at >= :cutoff
        ORDER  BY created_at ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "fraud_score", "risk_label", "decision_threshold",
        "latency_ms", "model_version", "created_at",
    ])
    df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce")
    df["risk_label"]  = pd.to_numeric(df["risk_label"],  errors="coerce").astype("Int64")
    df["latency_ms"]  = pd.to_numeric(df["latency_ms"],  errors="coerce")
    return df


def _fetch_daily_counts(engine, lookback_days: int) -> pd.DataFrame:
    """Return daily prediction + fraud counts."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    sql = text("""
        SELECT
            DATE(created_at AT TIME ZONE 'UTC')  AS day,
            COUNT(*)                              AS total_predictions,
            SUM(risk_label)                       AS fraud_predicted,
            AVG(fraud_score)                      AS avg_score
        FROM   prediction_log
        WHERE  created_at >= :cutoff
        GROUP  BY 1
        ORDER  BY 1
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["day", "total_predictions", "fraud_predicted", "avg_score"])
    df["fraud_rate"] = (
        pd.to_numeric(df["fraud_predicted"], errors="coerce")
        / df["total_predictions"].replace(0, np.nan)
    ).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Risk band classification (mirrors prediction_service.py)
# ---------------------------------------------------------------------------

def _score_to_band(score: float) -> str:
    if score < 0.25:  return "low"
    if score < 0.50:  return "medium"
    if score < 0.75:  return "high"
    return "critical"


# ---------------------------------------------------------------------------
# Alert builders
# ---------------------------------------------------------------------------

def _build_alerts(metrics: dict[str, Any]) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []

    live_fr   = metrics.get("fraud_prediction_rate")
    live_mean = metrics.get("mean_fraud_score")

    if live_fr is not None and live_fr > TRAIN_FRAUD_RATE * FRAUD_RATE_ALERT_MULT:
        alerts.append({
            "level":   "WARNING",
            "message": (
                f"Fraud prediction rate ({live_fr:.2%}) is more than "
                f"{FRAUD_RATE_ALERT_MULT}x the training baseline "
                f"({TRAIN_FRAUD_RATE:.2%}). Investigate label shift."
            ),
        })

    if live_mean is not None:
        score_delta = abs(live_mean - 0.05)
        if score_delta > SCORE_MEAN_ALERT_DELTA:
            alerts.append({
                "level":   "INFO",
                "message": (
                    f"Mean fraud score ({live_mean:.4f}) has shifted "
                    f"{score_delta:.4f} from expected baseline. "
                    "Consider running drift detection."
                ),
            })

    if metrics.get("total_predictions", 0) == 0:
        alerts.append({
            "level":   "INFO",
            "message": "No predictions found in the monitoring window. "
                       "Check that the API is receiving traffic.",
        })

    return alerts



# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def run_prediction_monitoring(
    report_dir:   Path = ROOT / "reports" / "monitoring_snapshots",
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Compute prediction distribution metrics over the last `lookback_days`
    and save a snapshot JSON to `report_dir`.

    Returns the snapshot dict.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    engine = _get_db_engine()

    logger.info("Loading predictions from last %d days …", lookback_days)
    df = _fetch_recent_predictions(engine, lookback_days)
    daily_df = _fetch_daily_counts(engine, lookback_days)

    n = len(df)
    logger.info("  Found %d predictions", n)

    if n == 0:
        metrics: dict[str, Any] = {
            "total_predictions": 0,
            "mean_fraud_score": None,
            "fraud_prediction_rate": None,
        }
    else:
        scores = df["fraud_score"].dropna().to_numpy(dtype=float)
        labels = df["risk_label"].dropna().to_numpy(dtype=int)

        bands = [_score_to_band(s) for s in scores]
        band_counts = {b: int(bands.count(b)) for b in ["low", "medium", "high", "critical"]}
        band_pct    = {b: round(band_counts[b] / n, 4) for b in band_counts}

        model_versions = df["model_version"].value_counts().to_dict()

        metrics = {
            "total_predictions":     n,
            "mean_fraud_score":      round(float(np.mean(scores)), 6) if len(scores) else None,
            "median_fraud_score":    round(float(np.median(scores)), 6) if len(scores) else None,
            "fraud_prediction_rate": round(float(labels.mean()), 6) if len(labels) else None,
            "score_percentiles": {
                "p50":  round(float(np.percentile(scores, 50)),  6),
                "p90":  round(float(np.percentile(scores, 90)),  6),
                "p95":  round(float(np.percentile(scores, 95)),  6),
                "p99":  round(float(np.percentile(scores, 99)),  6),
            } if len(scores) else {},
            "risk_band_counts":      band_counts,
            "risk_band_percentages": band_pct,
            "mean_latency_ms":       round(float(df["latency_ms"].mean()), 1) if n else None,
            "p99_latency_ms":        round(float(df["latency_ms"].quantile(0.99)), 1) if n else None,
            "model_version_counts":  {str(k): v for k, v in model_versions.items()},
        }

    # Baseline comparison
    metrics["baseline_fraud_rate"]           = TRAIN_FRAUD_RATE
    metrics["baseline_vs_live_fraud_rate_delta"] = (
        round((metrics.get("fraud_prediction_rate") or 0) - TRAIN_FRAUD_RATE, 6)
    )

    # Daily trend
    daily_trend: list[dict[str, Any]] = []
    if not daily_df.empty:
        for _, row in daily_df.iterrows():
            daily_trend.append({
                "day":               str(row["day"]),
                "total_predictions": int(row["total_predictions"]),
                "fraud_predicted":   int(row["fraud_predicted"] or 0),
                "avg_score":         round(float(row["avg_score"] or 0), 6),
                "fraud_rate":        round(float(row["fraud_rate"]), 6),
            })

    alerts = _build_alerts(metrics)

    snapshot: dict[str, Any] = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "lookback_days":  lookback_days,
        "metrics":        metrics,
        "daily_trend":    daily_trend,
        "alerts":         alerts,
    }

    # Save
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = report_dir / f"snapshot_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    logger.info("Monitoring snapshot saved: %s", out)

    latest = report_dir / "snapshot_latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    _log_summary(snapshot)
    return snapshot


def _log_summary(snap: dict[str, Any]) -> None:
    m = snap["metrics"]
    logger.info("=" * 55)
    logger.info("  PREDICTION MONITORING SUMMARY")
    logger.info("=" * 55)
    logger.info("  Period          : last %d days",  snap["lookback_days"])
    logger.info("  Total preds     : %s",            m.get("total_predictions", "N/A"))
    logger.info("  Mean score      : %s",            m.get("mean_fraud_score", "N/A"))
    logger.info("  Fraud pred rate : %s",            m.get("fraud_prediction_rate", "N/A"))
    logger.info("  Baseline rate   : %.4f",          m.get("baseline_fraud_rate", 0))
    logger.info("  Rate delta      : %+.4f",         m.get("baseline_vs_live_fraud_rate_delta", 0))
    for alert in snap.get("alerts", []):
        logger.warning("  [%s] %s", alert["level"], alert["message"])
    logger.info("=" * 55)
