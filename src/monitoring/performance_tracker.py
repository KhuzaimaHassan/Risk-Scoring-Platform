"""
src/monitoring/performance_tracker.py
---------------------------------------
Model performance tracking using ground-truth fraud labels.

This module joins prediction_log with fact_transaction to retrieve the
actual fraud_label for each prediction, then computes rolling performance
metrics over a configurable time window.

When is ground truth available?
--------------------------------
In this platform, fact_transaction.fraud_label is set at seeding time
(or asynchronously by an investigation team). The prediction_log.outcome
column is resolved via a reconciliation job.

This tracker queries the JOIN directly:
    prediction_log.transaction_id → fact_transaction.fraud_label

Metrics computed (when ground truth available)
----------------------------------------------
- Rolling AUC-ROC
- Precision / Recall / F1 at current threshold
- Confusion matrix counts (TP, TN, FP, FN)
- Daily performance series (for trend charts)

Output
------
Saves JSON to reports/monitoring_snapshots/performance_{ts}.json
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOOKBACK_DAYS:      int   = 30   # longer window for performance tracking
MIN_LABELLED_ROWS:  int   = 20   # skip AUC if fewer labelled rows available
TRAINING_THRESHOLD: float = 0.5  # default threshold (overridden by model meta)


def _get_db_engine():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("SYNC_DATABASE_URL", "")
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )
    return create_engine(db_url, pool_pre_ping=True)


def _fetch_ground_truth_predictions(engine, lookback_days: int) -> pd.DataFrame:
    """
    Join prediction_log with fact_transaction to get (fraud_score, actual_label).
    Only returns rows where fact_transaction.fraud_label IS NOT NULL.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    sql = text("""
        SELECT
            pl.log_id,
            pl.created_at,
            pl.fraud_score,
            pl.risk_label          AS predicted_label,
            pl.decision_threshold,
            pl.model_version,
            ft.fraud_label         AS actual_label
        FROM   prediction_log pl
        JOIN   fact_transaction ft
               ON ft.transaction_id = pl.transaction_id
        WHERE  pl.created_at  >= :cutoff
          AND  ft.fraud_label IS NOT NULL
        ORDER  BY pl.created_at ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "log_id", "created_at", "fraud_score", "predicted_label",
        "decision_threshold", "model_version", "actual_label",
    ])
    df["fraud_score"]     = pd.to_numeric(df["fraud_score"],     errors="coerce")
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce").astype("Int64")
    df["actual_label"]    = pd.to_numeric(df["actual_label"],    errors="coerce").astype("Int64")
    return df.dropna(subset=["fraud_score", "actual_label"])


def _compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute classification metrics at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    result: dict[str, Any] = {
        "n_samples":  n,
        "n_fraud":    n_pos,
        "n_legit":    n_neg,
        "threshold":  threshold,
    }

    if n_pos < 2 or n_neg < 2:
        result["roc_auc"]   = None
        result["precision"] = None
        result["recall"]    = None
        result["f1"]        = None
        result["note"]      = "Insufficient class diversity for metrics"
        return result

    try:
        result["roc_auc"]   = round(float(roc_auc_score(y_true, y_score)), 6)
    except ValueError:
        result["roc_auc"]   = None

    result["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 6)
    result["recall"]    = round(float(recall_score(y_true,    y_pred, zero_division=0)), 6)
    result["f1"]        = round(float(f1_score(y_true,        y_pred, zero_division=0)), 6)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        result["conf_matrix"] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

    return result


def _compute_daily_performance(
    df: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    """Compute per-day metrics for trend analysis."""
    df = df.copy()
    df["day"] = pd.to_datetime(df["created_at"]).dt.date
    daily: list[dict[str, Any]] = []

    for day, grp in df.groupby("day"):
        y_true  = grp["actual_label"].to_numpy(dtype=int)
        y_score = grp["fraud_score"].to_numpy(dtype=float)
        m = _compute_metrics(y_true, y_score, threshold)
        daily.append({
            "day":        str(day),
            "n_samples":  m["n_samples"],
            "n_fraud":    m["n_fraud"],
            "roc_auc":    m.get("roc_auc"),
            "precision":  m.get("precision"),
            "recall":     m.get("recall"),
            "f1":         m.get("f1"),
        })

    return daily


def _load_model_threshold(model_dir: Path = ROOT / "models") -> float:
    """Read production model threshold from registry.json."""
    registry_path = model_dir / "registry.json"
    if not registry_path.exists():
        return TRAINING_THRESHOLD
    try:
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
        versions: list[dict] = registry.get("versions", [])
        prod = next((v for v in reversed(versions) if v.get("stage") == "production"), None)
        if prod:
            return float(
                prod.get("metrics", {}).get("threshold_used", TRAINING_THRESHOLD)
            )
    except Exception:
        pass
    return TRAINING_THRESHOLD


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def run_performance_tracking(
    report_dir:    Path = ROOT / "reports" / "monitoring_snapshots",
    model_dir:     Path = ROOT / "models",
    lookback_days: int  = LOOKBACK_DAYS,
) -> dict[str, Any]:
    """
    Compute rolling model performance metrics by joining prediction_log
    with actual fraud labels from fact_transaction.

    Returns the performance report dict (also saved to disk).
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    engine    = _get_db_engine()
    threshold = _load_model_threshold(model_dir)

    logger.info("Loading ground-truth predictions (last %d days) …", lookback_days)
    df = _fetch_ground_truth_predictions(engine, lookback_days)
    logger.info("  Labelled predictions: %d", len(df))

    if len(df) < MIN_LABELLED_ROWS:
        logger.warning(
            "Only %d labelled rows available (min=%d). "
            "Performance metrics will be limited.",
            len(df), MIN_LABELLED_ROWS,
        )

    # ── Overall metrics ───────────────────────────────────────────────
    if df.empty:
        overall_metrics = {"note": "No labelled predictions available in window."}
        daily_series    = []
    else:
        y_true  = df["actual_label"].to_numpy(dtype=int)
        y_score = df["fraud_score"].to_numpy(dtype=float)
        overall_metrics = _compute_metrics(y_true, y_score, threshold)
        daily_series    = _compute_daily_performance(df, threshold)

    # ── Comparison to training baseline ──────────────────────────────
    baseline = {
        "roc_auc":   0.715748,
        "precision": 0.222222,
        "recall":    0.039474,
        "f1":        0.067039,
        "threshold": 0.5,
        "source":    "eval_v20260301_113051.json",
    }

    def _delta(key: str) -> float | None:
        live = overall_metrics.get(key)
        base = baseline.get(key)
        if live is None or base is None:
            return None
        return round(float(live) - float(base), 6)

    performance_delta = {
        "roc_auc_delta":   _delta("roc_auc"),
        "precision_delta": _delta("precision"),
        "recall_delta":    _delta("recall"),
        "f1_delta":        _delta("f1"),
    }

    # ── Per-version breakdown ─────────────────────────────────────────
    version_breakdown: list[dict[str, Any]] = []
    if not df.empty:
        for ver, grp in df.groupby("model_version"):
            y_t = grp["actual_label"].to_numpy(dtype=int)
            y_s = grp["fraud_score"].to_numpy(dtype=float)
            m   = _compute_metrics(y_t, y_s, threshold)
            version_breakdown.append({
                "model_version": str(ver),
                "n_samples":     m["n_samples"],
                "roc_auc":       m.get("roc_auc"),
                "f1":            m.get("f1"),
            })

    report: dict[str, Any] = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "lookback_days":     lookback_days,
        "threshold_used":    threshold,
        "overall_metrics":   overall_metrics,
        "performance_delta": performance_delta,
        "training_baseline": baseline,
        "daily_performance": daily_series,
        "version_breakdown": version_breakdown,
    }

    # ── Save ──────────────────────────────────────────────────────────
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = report_dir / f"performance_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Performance report saved: %s", out)

    latest = report_dir / "performance_latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _log_summary(report)
    return report


def _log_summary(report: dict[str, Any]) -> None:
    m     = report.get("overall_metrics", {})
    delta = report.get("performance_delta", {})
    logger.info("=" * 55)
    logger.info("  PERFORMANCE TRACKER SUMMARY")
    logger.info("=" * 55)
    logger.info("  Period      : last %d days", report["lookback_days"])
    logger.info("  Threshold   : %.4f",         report["threshold_used"])
    logger.info("  Labelled    : %s",            m.get("n_samples", "N/A"))
    logger.info("  AUC-ROC     : %s  (delta %s)",
                m.get("roc_auc", "N/A"), delta.get("roc_auc_delta", "N/A"))
    logger.info("  Precision   : %s  (delta %s)",
                m.get("precision", "N/A"), delta.get("precision_delta", "N/A"))
    logger.info("  Recall      : %s  (delta %s)",
                m.get("recall", "N/A"), delta.get("recall_delta", "N/A"))
    logger.info("  F1          : %s  (delta %s)",
                m.get("f1", "N/A"), delta.get("f1_delta", "N/A"))
    logger.info("=" * 55)
