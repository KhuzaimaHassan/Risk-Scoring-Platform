"""
scripts/run_monitoring.py
--------------------------
CLI entrypoint for the monitoring & drift detection pipeline.

Runs all three monitoring checks in sequence:
    1. drift_detection     — KS / Chi-square test per feature
    2. prediction_monitor  — Fraud rate, score distribution
    3. performance_tracker — Rolling AUC / precision / recall

Usage
-----
    # Run all monitors (default: last 7 days drift, 7 days preds, 30 days perf)
    python scripts/run_monitoring.py

    # Custom windows
    python scripts/run_monitoring.py --drift-days 14 --perf-days 60

    # Run only specific modules
    python scripts/run_monitoring.py --skip-drift
    python scripts/run_monitoring.py --skip-performance

Output
------
    reports/drift_reports/drift_{timestamp}.json
    reports/drift_reports/drift_latest.json
    reports/monitoring_snapshots/snapshot_{timestamp}.json
    reports/monitoring_snapshots/snapshot_latest.json
    reports/monitoring_snapshots/performance_{timestamp}.json
    reports/monitoring_snapshots/performance_latest.json

Scheduling as a cron job (Linux/macOS)
---------------------------------------
    # Every day at 06:00 UTC
    0 6 * * * cd /path/to/risk-scoring-platform && \
        ./venv/bin/python scripts/run_monitoring.py >> logs/monitoring.log 2>&1

Windows Task Scheduler
-----------------------
    Action:  python.exe scripts\\run_monitoring.py
    Start in: C:\\path\\to\\risk-scoring-platform
    Schedule: Daily, 06:00
    Redirect stdout to: logs\\monitoring.log
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("monitoring")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run monitoring & drift detection for the Risk Scoring Platform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--drift-days",  type=int, default=7,
                        help="Lookback window (days) for drift detection")
    parser.add_argument("--pred-days",   type=int, default=7,
                        help="Lookback window (days) for prediction monitoring")
    parser.add_argument("--perf-days",   type=int, default=30,
                        help="Lookback window (days) for performance tracking")
    parser.add_argument("--skip-drift",       action="store_true", help="Skip drift detection")
    parser.add_argument("--skip-prediction",  action="store_true", help="Skip prediction monitor")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tracker")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports",
                        help="Base directory for all report outputs")
    return parser.parse_args()


def section(title: str) -> None:
    logger.info("")
    logger.info("=" * 55)
    logger.info("  %s", title)
    logger.info("=" * 55)


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    logger.info("Risk Scoring Platform — Monitoring Run")
    logger.info("Report directory: %s", args.report_dir)

    results: dict[str, str] = {}

    # ── 1. Drift Detection ─────────────────────────────────────────────
    if not args.skip_drift:
        section("DRIFT DETECTION")
        t0 = time.perf_counter()
        try:
            from src.monitoring.drift_detection import run_drift_detection
            report = run_drift_detection(
                report_dir  = args.report_dir / "drift_reports",
                lookback_days = args.drift_days,
            )
            elapsed = time.perf_counter() - t0
            alert   = report.get("alert_level", "unknown").upper()
            drifted = report.get("features_drifted", 0)
            results["drift"] = f"DONE [{alert}] — {drifted} features drifted ({elapsed:.1f}s)"
        except Exception as exc:
            logger.exception("Drift detection failed: %s", exc)
            results["drift"] = f"FAILED — {exc}"
    else:
        results["drift"] = "SKIPPED"

    # ── 2. Prediction Monitoring ────────────────────────────────────────
    if not args.skip_prediction:
        section("PREDICTION MONITORING")
        t0 = time.perf_counter()
        try:
            from src.monitoring.prediction_monitor import run_prediction_monitoring
            snap = run_prediction_monitoring(
                report_dir    = args.report_dir / "monitoring_snapshots",
                lookback_days = args.pred_days,
            )
            elapsed = time.perf_counter() - t0
            n       = snap["metrics"].get("total_predictions", 0)
            fr      = snap["metrics"].get("fraud_prediction_rate")
            fr_str  = f"{fr:.2%}" if fr is not None else "N/A"
            results["prediction"] = (
                f"DONE — {n} preds, fraud_rate={fr_str} ({elapsed:.1f}s)"
            )
        except Exception as exc:
            logger.exception("Prediction monitoring failed: %s", exc)
            results["prediction"] = f"FAILED — {exc}"
    else:
        results["prediction"] = "SKIPPED"

    # ── 3. Performance Tracking ─────────────────────────────────────────
    if not args.skip_performance:
        section("PERFORMANCE TRACKING")
        t0 = time.perf_counter()
        try:
            from src.monitoring.performance_tracker import run_performance_tracking
            perf = run_performance_tracking(
                report_dir    = args.report_dir / "monitoring_snapshots",
                lookback_days = args.perf_days,
            )
            elapsed = time.perf_counter() - t0
            auc     = perf["overall_metrics"].get("roc_auc")
            auc_str = f"{auc:.4f}" if auc is not None else "N/A (insufficient labels)"
            results["performance"] = f"DONE — AUC-ROC={auc_str} ({elapsed:.1f}s)"
        except Exception as exc:
            logger.exception("Performance tracking failed: %s", exc)
            results["performance"] = f"FAILED — {exc}"
    else:
        results["performance"] = "SKIPPED"

    # ── Summary ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    logger.info("")
    logger.info("=" * 55)
    logger.info("  MONITORING RUN COMPLETE  (%.1fs total)", total_elapsed)
    logger.info("=" * 55)
    for module, status in results.items():
        logger.info("  %-14s: %s", module.upper(), status)
    logger.info("=" * 55)
    logger.info("  Reports saved to: %s", args.report_dir)
    logger.info("=" * 55)

    # Exit 1 if any module failed
    if any("FAILED" in s for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
