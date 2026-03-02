"""
src/training/save_model.py
----------------------------
Model artifact persistence and loading utilities.

Responsibility:
    Serialize and deserialize sklearn model objects plus all associated
    metadata required for reproducible inference and registry tracking.

Artifacts produced per training run:
    models/
    ├── model_{version}.pkl          — Joblib-serialized sklearn pipeline/estimator
    ├── metadata_{version}.json      — Training config, metrics, feature list
    └── (symlink) model_active.pkl   — Points to current production model

    reports/
    └── eval_{version}.json          — Full evaluation report (from evaluate.py)

Design rules:
    - load_model() is called at FastAPI startup — must be fast and reliable.
    - save_model() is called once per training run — correctness > speed.
    - Feature column list is always saved WITH the model for version safety.
    - Never import DB or API modules here.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _artifact_dir(base_dir: Path) -> Path:
    """Ensure the models/ directory exists and return it."""
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def model_artifact_path(base_dir: Path, version: str) -> Path:
    """Return the filesystem path for a model pickle file."""
    return _artifact_dir(base_dir) / f"model_{version}.pkl"


def metadata_path(base_dir: Path, version: str) -> Path:
    """Return the filesystem path for a model metadata JSON file."""
    return _artifact_dir(base_dir) / f"metadata_{version}.json"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    version: str,
    feature_columns: list[str],
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    base_dir: Path,
    training_rows: int,
    training_duration_s: float,
    description: str = "",
) -> tuple[Path, Path]:
    """
    Persist a fitted model artifact and its metadata JSON.

    Args:
        model:               Fitted sklearn estimator or Pipeline.
        version:             Unique version tag, e.g. "v3_20260301_143022".
        feature_columns:     Ordered feature list the model was trained on.
        hyperparameters:     Dict of hyperparameter values used during training.
        metrics:             Evaluation metrics dict from evaluate.py.
        base_dir:            Root directory for model artifacts (models/).
        training_rows:       Number of training samples.
        training_duration_s: Wall-clock training time in seconds.
        description:         Optional human-readable release notes.

    Returns:
        Tuple of (artifact_path, metadata_json_path).

    Raises:
        IOError: If serialisation fails (e.g. disk full).
    """
    artifact_path = model_artifact_path(base_dir, version)
    meta_path     = metadata_path(base_dir, version)

    # --- Serialize model ---
    logger.info("Saving model artifact → %s", artifact_path)
    joblib.dump(model, artifact_path, compress=3)   # lz4 compression (fast)
    artifact_size = artifact_path.stat().st_size
    logger.info("  Artifact size: %.2f MB", artifact_size / 1_048_576)

    # --- Write metadata JSON ---
    metadata: dict[str, Any] = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "framework": "sklearn",
        "framework_version": _sklearn_version(),
        "python_version": platform.python_version(),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_size_bytes": artifact_size,
        "training_rows": training_rows,
        "training_duration_s": round(training_duration_s, 3),
        "hyperparameters": hyperparameters,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        # Store only the summary metrics (not raw ROC curve data)
        "metrics": {
            k: v for k, v in metrics.get("default_threshold_metrics", metrics).items()
            if k not in ("classification_report", "roc_curve", "feature_importances")
        },
        "stage": "candidate",    # Promoted to production via model_registry
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metadata saved → %s", meta_path)

    return artifact_path, meta_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model(artifact_path: Path) -> Any:
    """
    Load a serialized sklearn model from disk.

    This is the function called by the FastAPI lifespan handler at startup
    and by the retraining pipeline when validating newly trained models.

    Args:
        artifact_path: Full path to the .pkl model file.

    Returns:
        Fitted sklearn estimator / Pipeline.

    Raises:
        FileNotFoundError: If the artifact_path does not exist.
        RuntimeError:      If joblib.load raises for any reason.
    """
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {artifact_path}\n"
            f"Run `python scripts/train_model.py` to produce a trained model."
        )
    logger.info("Loading model artifact from %s …", artifact_path)
    try:
        model = joblib.load(artifact_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to deserialise model: {exc}") from exc
    logger.info("  Model type: %s", type(model).__name__)
    return model


def load_metadata(meta_path: Path) -> dict[str, Any]:
    """
    Load model metadata JSON from disk.

    Args:
        meta_path: Full path to the metadata_{version}.json file.

    Returns:
        Metadata dict (same structure as written by save_model).

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_with_metadata(
    version: str,
    base_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """
    Convenience function — load both model artifact and metadata in one call.

    Used by the prediction layer at startup:
        model, meta = load_model_with_metadata("v3_20260301", Path("models/"))
        feature_columns = meta["feature_columns"]
        threshold = meta["metrics"].get("threshold_used", 0.50)

    Returns:
        Tuple of (fitted_model, metadata_dict).
    """
    art_path  = model_artifact_path(base_dir, version)
    meta_path = metadata_path(base_dir, version)
    model     = load_model(art_path)
    meta      = load_metadata(meta_path)
    return model, meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sklearn_version() -> str:
    """Return the installed scikit-learn version string."""
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return "unknown"


def generate_version_tag() -> str:
    """
    Generate a timestamped, monotonically increasing version tag.

    Format: v{YYYYMMDD}_{HHMMSS}
    Example: v20260301_143022

    The timestamp-based tag guarantees lexicographic ordering == chronological
    ordering, making it trivial to find the latest version by sorting strings.
    """
    now = datetime.now(timezone.utc)
    return f"v{now.strftime('%Y%m%d_%H%M%S')}"
