"""
src/training/train.py
-----------------------
Main training orchestrator for the fraud detection model.

Training Flow:
    1.  Load training config from config/training.yaml
    2.  Build feature matrix via FeaturePipeline.build_training_dataset()
    3.  Perform time-aware train/test split (chronological — no shuffle)
    4.  Build sklearn Pipeline: preprocessing → RandomForestClassifier
    5.  Fit on training split
    6.  Evaluate on held-out test split via evaluate.py
    7.  Persist model artifact and metadata via save_model.py
    8.  Register new version in model registry

Retraining compatibility:
    The same train() function is called by:
        - scripts/train_model.py   (initial / manual training)
        - src/retraining/pipeline.py (automated retraining trigger)

    Both paths accept an optional `TrainConfig` override so that the
    retraining pipeline can inject a different dataset window without
    modifying config/training.yaml.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_pipeline import FEATURE_COLUMNS, FeaturePipeline
from src.training.evaluate import evaluate_model, save_evaluation_report
from src.training.model_registry import ModelRegistry
from src.training.save_model import generate_version_tag, save_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    All training hyperparameters and data settings in one place.

    Loaded from config/training.yaml by default; can be overridden
    programmatically (e.g. during retraining with different time windows).
    """
    # Model identity
    model_name: str = "fraud_classifier"

    # RandomForest hyperparameters
    n_estimators: int = 300
    max_depth: int | None = 20
    min_samples_split: int = 10
    min_samples_leaf: int = 4
    max_features: str = "sqrt"
    class_weight: str = "balanced"
    n_jobs: int = -1
    random_state: int = 42
    bootstrap: bool = True
    oob_score: bool = True

    # Split strategy
    split_strategy: str = "time"   # "time" | "random"
    test_size: float = 0.20

    # Decision threshold
    decision_threshold: float = 0.50
    optimise_threshold_for: str = "f1"   # "f1" | "recall" | "precision"

    # Training data window (None = all labelled data)
    train_since: datetime | None = None
    train_until: datetime | None = None

    # Paths
    model_artifact_dir: Path = field(default_factory=lambda: Path("models"))
    report_dir: Path = field(default_factory=lambda: Path("reports"))
    config_path: Path = field(default_factory=lambda: Path("config/training.yaml"))

    # Registry behaviour
    auto_promote: bool = False       # If True, auto-promote to production on success

    # Description / notes for this run
    description: str = ""


def load_train_config(config_path: Path | str = "config/training.yaml") -> TrainConfig:
    """
    Load TrainConfig from a YAML file.

    Only fields that match TrainConfig's attributes are applied.
    Unknown YAML keys are silently ignored (forward-compatibility).

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Populated TrainConfig dataclass.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(
            "Training config not found at '%s'. Using all defaults.", config_path
        )
        return TrainConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = TrainConfig()

    # Top-level model section
    model_section = raw.get("model", {})
    cfg.model_name = model_section.get("name", cfg.model_name)

    # RandomForest section
    rf = raw.get("random_forest", {})
    if "n_estimators"     in rf: cfg.n_estimators      = int(rf["n_estimators"])
    if "max_depth"        in rf: cfg.max_depth          = rf["max_depth"]       # May be null
    if "min_samples_split"in rf: cfg.min_samples_split  = int(rf["min_samples_split"])
    if "min_samples_leaf" in rf: cfg.min_samples_leaf   = int(rf["min_samples_leaf"])
    if "max_features"     in rf: cfg.max_features       = str(rf["max_features"])
    if "class_weight"     in rf: cfg.class_weight       = str(rf["class_weight"])
    if "n_jobs"           in rf: cfg.n_jobs             = int(rf["n_jobs"])
    if "random_state"     in rf: cfg.random_state       = int(rf["random_state"])
    if "bootstrap"        in rf: cfg.bootstrap          = bool(rf["bootstrap"])
    if "oob_score"        in rf: cfg.oob_score          = bool(rf["oob_score"])

    # Split section
    split = raw.get("split", {})
    if "strategy"  in split: cfg.split_strategy = str(split["strategy"])
    if "test_size" in split: cfg.test_size      = float(split["test_size"])

    # Thresholds section
    thresh = raw.get("thresholds", {})
    if "decision" in thresh:
        cfg.decision_threshold = float(thresh["decision"])
    if "precision_recall_tradeoff" in thresh:
        cfg.optimise_threshold_for = str(thresh["precision_recall_tradeoff"])

    # Retraining section
    retrain = raw.get("retraining", {})
    if "auto_promote" in retrain:
        cfg.auto_promote = bool(retrain["auto_promote"])

    logger.info("TrainConfig loaded from '%s'.", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def time_aware_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological train/test split — NO shuffling.

    The last `test_size` fraction of rows (by row order, which should be
    chronological after FeaturePipeline returns sorted data) forms the test
    set. This mirrors how the model is deployed: trained on historical data,
    evaluated (and ultimately used) on future data.

    Args:
        X:         Feature DataFrame (rows in ascending time order).
        y:         Target Series aligned with X.
        test_size: Fraction of total samples to allocate to test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1). Got: {test_size}")

    n = len(X)
    split_idx = int(n * (1 - test_size))

    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test  = X.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test  = y.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "Time-aware split: train=%d rows (%.0f%%), test=%d rows (%.0f%%)",
        len(X_train), (1 - test_size) * 100,
        len(X_test), test_size * 100,
    )
    logger.info(
        "  Train fraud rate: %.2f%%  |  Test fraud rate: %.2f%%",
        y_train.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Standard shuffled train/test split (used when split_strategy='random').

    Note: Not recommended for time-series fraud data — prefer time_aware_split.
    Exposed here for experimentation / comparison only.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Random split (stratified): train=%d, test=%d", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model_pipeline(cfg: TrainConfig) -> Pipeline:
    """
    Build a sklearn Pipeline with:
      1. SimpleImputer  — median fallback for any remaining NaN
      2. StandardScaler — normalises continuous features (helps tree stability)
      3. RandomForestClassifier — the core estimator

    Using a Pipeline ensures that all preprocessing steps are bundled
    with the model artifact, preventing training-serving skew.

    Args:
        cfg: TrainConfig with all hyperparameter values.

    Returns:
        Unfitted sklearn Pipeline.
    """
    rf_params: dict[str, Any] = {
        "n_estimators":      cfg.n_estimators,
        "max_depth":         cfg.max_depth,
        "min_samples_split": cfg.min_samples_split,
        "min_samples_leaf":  cfg.min_samples_leaf,
        "max_features":      cfg.max_features,
        "class_weight":      cfg.class_weight,
        "n_jobs":            cfg.n_jobs,
        "random_state":      cfg.random_state,
        "bootstrap":         cfg.bootstrap,
        "oob_score":         cfg.oob_score,
    }

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("classifier", RandomForestClassifier(**rf_params)),
    ])

    logger.info(
        "Model pipeline built: RandomForest(n_estimators=%d, max_depth=%s, "
        "class_weight=%s, n_jobs=%d)",
        cfg.n_estimators, cfg.max_depth, cfg.class_weight, cfg.n_jobs,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    cfg: TrainConfig | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """
    Execute one full training run:
        load data → split → build model → fit → evaluate → save → register.

    This function is idempotent for the same `version` tag IF the version
    does not already exist in the registry. Subsequent calls with the same
    version will raise a ValueError from ModelRegistry.register().

    Args:
        cfg:     Training configuration. If None, loads from training.yaml.
        version: Version tag to use (e.g. "v20260301_143022"). If None,
                 auto-generates a timestamp-based tag.

    Returns:
        Dict with keys:
            version          — The version tag used.
            artifact_path    — Path to the saved .pkl file.
            metrics          — Evaluation metrics dict.
            promoted         — True if auto-promoted to production.
            registry_summary — ModelRegistry.summary() dict.
    """
    if cfg is None:
        cfg = load_train_config()

    if version is None:
        version = generate_version_tag()

    logger.info("=" * 60)
    logger.info("Training run | version=%s | model=%s", version, cfg.model_name)
    logger.info("=" * 60)

    t_start = time.perf_counter()

    # ── Step 1: Load feature matrix ─────────────────────────────────────
    logger.info("[1/6] Loading feature matrix from FeaturePipeline …")
    pipeline = FeaturePipeline(lookback_days=7)
    X, y = pipeline.build_training_dataset(
        since=cfg.train_since,
        until=cfg.train_until,
    )

    if len(X) == 0:
        raise RuntimeError(
            "No labelled training data found. "
            "Seed the database first: python scripts/seed_db.py"
        )

    if y.sum() == 0:
        raise RuntimeError(
            "All training labels are 0 (no fraud samples). "
            "Check that fraud_label was injected correctly during seeding."
        )

    logger.info(
        "  Loaded %d samples | %d features | %.2f%% fraud",
        len(X), len(FEATURE_COLUMNS), y.mean() * 100,
    )

    # ── Step 2: Train/test split ─────────────────────────────────────────
    logger.info("[2/6] Splitting dataset (strategy='%s', test_size=%.2f) …",
                cfg.split_strategy, cfg.test_size)

    if cfg.split_strategy == "time":
        X_train, X_test, y_train, y_test = time_aware_split(X, y, cfg.test_size)
    elif cfg.split_strategy == "random":
        X_train, X_test, y_train, y_test = random_split(
            X, y, cfg.test_size, cfg.random_state
        )
    else:
        raise ValueError(f"Unknown split_strategy: '{cfg.split_strategy}'")

    # ── Step 3: Build model pipeline ─────────────────────────────────────
    logger.info("[3/6] Building model pipeline …")
    model = build_model_pipeline(cfg)

    # ── Step 4: Fit ─────────────────────────────────────────────────────
    logger.info(
        "[4/6] Fitting RandomForestClassifier on %d samples …", len(X_train)
    )
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_elapsed = time.perf_counter() - fit_start
    training_duration_s = time.perf_counter() - t_start

    # OOB score from the underlying RF (accessible in sklearn Pipeline)
    rf_estimator = model.named_steps["classifier"]
    if hasattr(rf_estimator, "oob_score_"):
        logger.info("  OOB score (training set): %.4f", rf_estimator.oob_score_)

    logger.info("  Fit completed in %.1f seconds.", fit_elapsed)

    # ── Step 5: Evaluate ─────────────────────────────────────────────────
    logger.info("[5/6] Evaluating on %d test samples …", len(X_test))
    eval_report = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_columns=FEATURE_COLUMNS,
        threshold=cfg.decision_threshold,
        optimise_threshold_for=cfg.optimise_threshold_for,
    )

    # Persist evaluation report
    report_path = cfg.report_dir / f"eval_{version}.json"
    save_evaluation_report(eval_report, report_path)

    # ── Step 6: Save model artifact ──────────────────────────────────────
    logger.info("[6/6] Saving model artifact and registering …")

    # Collect hyperparameters for metadata
    hyperparameters: dict[str, Any] = {
        "n_estimators":      cfg.n_estimators,
        "max_depth":         cfg.max_depth,
        "min_samples_split": cfg.min_samples_split,
        "min_samples_leaf":  cfg.min_samples_leaf,
        "max_features":      cfg.max_features,
        "class_weight":      cfg.class_weight,
        "random_state":      cfg.random_state,
        "bootstrap":         cfg.bootstrap,
        "split_strategy":    cfg.split_strategy,
        "test_size":         cfg.test_size,
        "decision_threshold": cfg.decision_threshold,
    }

    # Summary metrics for metadata JSON (exclude nested objects)
    summary_metrics = {
        k: v for k, v in
        eval_report.get("default_threshold_metrics", {}).items()
        if k not in ("classification_report", "confusion_matrix")
    }
    # Add confusion matrix as flat fields
    cm = eval_report.get("default_threshold_metrics", {}).get("confusion_matrix", {})
    summary_metrics.update({f"cm_{k}": v for k, v in cm.items()})

    artifact_path, meta_path = save_model(
        model=model,
        version=version,
        feature_columns=FEATURE_COLUMNS,
        hyperparameters=hyperparameters,
        metrics=summary_metrics,
        base_dir=cfg.model_artifact_dir,
        training_rows=len(X_train),
        training_duration_s=training_duration_s,
        description=cfg.description,
    )

    # ── Register in model registry ───────────────────────────────────────
    registry = ModelRegistry(base_dir=cfg.model_artifact_dir)

    import json as _json
    with open(meta_path, "r") as f:
        metadata = _json.load(f)

    registry.register(metadata)

    # Auto-promote to production if configured
    promoted = False
    if cfg.auto_promote:
        registry.promote_model(version, stage="production", promoted_by="train.py[auto]")
        promoted = True
        logger.info("Auto-promoted '%s' to production.", version)
    else:
        logger.info(
            "Model registered as CANDIDATE. Promote manually:\n"
            "  from src.training.model_registry import ModelRegistry\n"
            "  ModelRegistry(Path('models/')).promote_model('%s')", version
        )

    total_elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("Training complete in %.1f seconds.", total_elapsed)
    logger.info("  Version    : %s", version)
    logger.info("  AUC-ROC    : %.4f", summary_metrics.get("roc_auc", 0))
    logger.info("  AUC-PR     : %.4f", summary_metrics.get("pr_auc", 0))
    logger.info("  F1         : %.4f", summary_metrics.get("f1", 0))
    logger.info("  Recall     : %.4f", summary_metrics.get("recall", 0))
    logger.info("  Precision  : %.4f", summary_metrics.get("precision", 0))
    logger.info("  Artifact   : %s", artifact_path)
    logger.info("=" * 60)

    return {
        "version": version,
        "artifact_path": str(artifact_path),
        "metrics": summary_metrics,
        "eval_report_path": str(report_path),
        "promoted": promoted,
        "registry_summary": registry.summary(),
    }
