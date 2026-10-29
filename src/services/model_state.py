"""Publish a complete immutable model bundle in one assignment."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from src.features.feature_pipeline import FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS
from config.settings import get_settings
from src.training.model_registry import ModelRegistry
from src.training.save_model import load_model_with_metadata

@dataclass(frozen=True)
class ModelBundle:
    model: Any
    metadata: dict
    version: str
    threshold: float

def load_bundle(directory: Path) -> ModelBundle:
    version = ModelRegistry(directory).get_latest_model()["version"]
    model, meta = load_model_with_metadata(version, directory)
    if meta.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise RuntimeError("Incompatible feature schema: retraining required")
    if meta.get("feature_columns") != FEATURE_COLUMNS or list(model.feature_names_in_) != FEATURE_COLUMNS:
        raise RuntimeError("Model feature columns do not match serving contract")
    settings = get_settings()
    if meta["metrics"].get("recall", 0) < settings.model_min_recall or meta["metrics"].get("f1", 0) < settings.model_min_f1:
        raise RuntimeError("Model does not meet serving quality gates")
    threshold = float(meta["metrics"]["threshold_used"])
    if not 0 < threshold < 1:
        raise RuntimeError("Invalid model threshold")
    model.named_steps["classifier"].n_jobs = 1
    return ModelBundle(model, meta, version, threshold)
