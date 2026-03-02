"""
src/training/evaluate.py
--------------------------
Model evaluation module — computes a comprehensive set of classification
metrics for the fraud detection model.

Responsibility:
    Receive ground-truth labels and predicted probabilities / binary labels,
    return a structured metrics dictionary, and optionally persist an
    evaluation report as JSON alongside the model artifact.

Design rules:
    - Pure functions — no side effects, no DB or file I/O inside metric fns.
    - All I/O is done via evaluate_and_report() which callers opt into.
    - Handles class imbalance gracefully (uses `average='binary'` throughout).
    - Finds the optimal threshold that maximises a configurable target metric.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Compute the full suite of binary classification metrics.

    Args:
        y_true:    Ground-truth binary labels (0 = legitimate, 1 = fraud).
        y_prob:    Predicted fraud probabilities in [0.0, 1.0].
        threshold: Decision threshold to convert probabilities to labels.

    Returns:
        Metrics dict with the following keys:
            roc_auc           — Area under the ROC curve
            pr_auc            — Area under the Precision-Recall curve
            f1                — F1 score at the specified threshold
            precision         — Precision at the specified threshold
            recall            — Recall at the specified threshold
            specificity       — True Negative Rate (1 - FPR)
            threshold_used    — The threshold that was applied
            confusion_matrix  — {tp, tn, fp, fn} dict
            support_fraud     — Number of fraud samples in y_true
            support_legit     — Number of legitimate samples in y_true
            class_imbalance_ratio — legit / fraud count ratio
            classification_report — sklearn text report (for logging)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    # --- ROC-AUC ---
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    # --- PR-AUC (more informative for imbalanced datasets) ---
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))

    # --- Threshold-dependent metrics ---
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall    = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1        = float(f1_score(y_true_arr, y_pred, zero_division=0))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true_arr, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, cm[0, 0])
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # --- Support ---
    n_fraud = int(y_true_arr.sum())
    n_legit = int(len(y_true_arr) - n_fraud)
    imbalance_ratio = float(n_legit / n_fraud) if n_fraud > 0 else float("inf")

    # --- Classification report (sklearn formatted string) ---
    cls_report = classification_report(
        y_true_arr, y_pred,
        target_names=["legitimate", "fraud"],
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "threshold_used": threshold,
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "support_fraud": n_fraud,
        "support_legit": n_legit,
        "class_imbalance_ratio": round(imbalance_ratio, 2),
        "classification_report": cls_report,
    }
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimise_for: str = "f1",
) -> tuple[float, float]:
    """
    Find the decision threshold that maximises a target metric.

    Args:
        y_true:       Ground-truth binary labels.
        y_prob:       Predicted fraud probabilities.
        optimise_for: Metric to maximise: "f1" | "recall" | "precision".
                      "recall" is recommended for fraud detection where missing
                      a fraud (FN) is costlier than a false alert (FP).

    Returns:
        Tuple of (optimal_threshold, optimal_metric_value).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)

    if optimise_for == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_scores = np.where(
                (precisions + recalls) > 0,
                2 * (precisions * recalls) / (precisions + recalls),
                0.0,
            )
        best_idx = int(np.argmax(f1_scores[:-1]))
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    elif optimise_for == "recall":
        # Max recall for precision >= 0.30 (minimum viable precision for fraud ops)
        valid_mask = precisions[:-1] >= 0.30
        if valid_mask.any():
            best_idx = int(np.argmax(recalls[:-1][valid_mask]))
            eligible_thresholds = thresholds[valid_mask]
            return float(eligible_thresholds[best_idx]), float(recalls[:-1][valid_mask][best_idx])
        # Fallback: return threshold with highest recall overall
        best_idx = int(np.argmax(recalls[:-1]))
        return float(thresholds[best_idx]), float(recalls[best_idx])

    elif optimise_for == "precision":
        best_idx = int(np.argmax(precisions[:-1]))
        return float(thresholds[best_idx]), float(precisions[best_idx])

    else:
        raise ValueError(f"optimise_for must be 'f1', 'recall', or 'precision'. Got: {optimise_for!r}")


def compute_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, list[float]]:
    """
    Compute ROC curve points for serialisation into evaluation report.

    Returns sampled points (max 200) for compact JSON storage.
    """
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_prob))

    # Downsample to 200 points for compact storage
    step = max(1, len(fpr) // 200)
    return {
        "fpr": [round(float(x), 6) for x in fpr[::step]],
        "tpr": [round(float(x), 6) for x in tpr[::step]],
    }


def compute_feature_importances(
    model: Any,
    feature_columns: list[str],
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """
    Extract feature importances from a fitted tree-based model.

    Works with sklearn RandomForest (feature_importances_ attribute).
    Sorted by importance descending, returns top_n features.

    Args:
        model:           Fitted sklearn estimator with feature_importances_.
        feature_columns: Ordered list of feature names (same order as training X).
        top_n:           Number of top features to return.

    Returns:
        List of dicts: [{"feature": str, "importance": float}, ...]
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not expose feature_importances_. Skipping.")
        return []

    importances = model.feature_importances_
    paired = sorted(
        zip(feature_columns, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance": round(float(imp), 8)}
        for name, imp in paired[:top_n]
    ]


# ---------------------------------------------------------------------------
# Composite evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
    threshold: float = 0.50,
    optimise_threshold_for: str = "f1",
) -> dict[str, Any]:
    """
    Run the full evaluation suite against a held-out test set.

    This is the primary function called by train.py after fitting the model.

    Args:
        model:                  Fitted sklearn estimator.
        X_test:                 Feature DataFrame (test split).
        y_test:                 Ground-truth labels (test split).
        feature_columns:        Ordered feature name list.
        threshold:              Initial decision threshold (from config).
        optimise_threshold_for: Metric to use for optimal threshold search.

    Returns:
        Comprehensive evaluation dict suitable for JSON serialisation.
    """
    logger.info("Running model evaluation on %d test samples …", len(X_test))

    y_prob  = model.predict_proba(X_test)[:, 1]   # Fraud class probability
    y_true  = y_test.values

    # Default-threshold metrics
    default_metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)

    # Optimal-threshold metrics
    opt_threshold, opt_metric_value = find_optimal_threshold(
        y_true, y_prob, optimise_for=optimise_threshold_for
    )
    optimal_metrics = compute_binary_metrics(y_true, y_prob, threshold=opt_threshold)
    optimal_metrics["optimised_for"] = optimise_threshold_for
    optimal_metrics["optimal_threshold"] = opt_threshold

    # ROC curve data
    roc_data = compute_roc_curve_data(y_true, y_prob)

    # OOB score (if model supports it)
    oob_score: float | None = None
    if hasattr(model, "oob_score_"):
        oob_score = round(float(model.oob_score_), 6)
        logger.info("  OOB score: %.4f", oob_score)

    # Feature importances
    importances = compute_feature_importances(model, feature_columns, top_n=20)

    # Summary log
    m = default_metrics
    logger.info(
        "Evaluation @ threshold=%.2f | AUC-ROC=%.4f | AUC-PR=%.4f | "
        "F1=%.4f | Precision=%.4f | Recall=%.4f",
        threshold, m["roc_auc"], m["pr_auc"], m["f1"], m["precision"], m["recall"],
    )
    logger.info(
        "Optimal threshold=%.4f (optimised for %s) | F1=%.4f | Recall=%.4f",
        opt_threshold, optimise_threshold_for,
        optimal_metrics["f1"], optimal_metrics["recall"],
    )
    logger.info("\n%s", m["classification_report"])

    evaluation_report: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_samples": len(X_test),
        "default_threshold_metrics": {
            k: v for k, v in default_metrics.items()
            if k != "classification_report"
        },
        "optimal_threshold_metrics": {
            k: v for k, v in optimal_metrics.items()
            if k != "classification_report"
        },
        "oob_score": oob_score,
        "roc_curve": roc_data,
        "feature_importances": importances,
        "classification_report_text": default_metrics["classification_report"],
    }
    return evaluation_report


def save_evaluation_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Persist the evaluation report as a JSON file alongside the model artifact.

    Args:
        report:      Evaluation dict from evaluate_model().
        output_path: Full path for the JSON file (e.g. reports/eval_v2_....json).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Evaluation report saved → %s", output_path)
